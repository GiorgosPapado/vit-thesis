"""ViT training (Google pretrained fine-tune / small ViT from scratch); needs torch + transformers.

Mirrors ``vit_google_10gest.ipynb`` and ``vit_train_from_scratch_10gest.ipynb``. Per epoch: mean train
loss over batches, train accuracy over the epoch, mean per-batch validation loss, validation accuracy.
The checkpoint is chosen on VALIDATION accuracy (ties: lower validation loss); the test set is evaluated
exactly once, after training, with that checkpoint.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math

import numpy as np
import yaml

from .augment import augment_data
from .data import (
    EMGDataset,
    channel_stats,
    extract_group_from_filename,
    extract_label_from_filename,
    list_mat_files,
    remap_labels,
    split_files_info,
    split_report,
)
from .evaluate import report_eval
from .seed import data_generator, seed_worker, set_seed
from .tracking import Tracker

log = logging.getLogger("emg_gesture")

MODES = ("google", "scratch")
GOOGLE_NAME = "google/vit-base-patch16-224-in21k"


def build_vit(mode: str, n_classes: int, model_cfg: dict | None = None):
    """Return a ``ViTForImageClassification``.

    google: ``from_pretrained(model_cfg['name'], num_labels=n_classes)`` (needs network/cache); with
    ``pretrained: false`` a randomly initialised model from the remaining keys is built instead (smoke tests).
    scratch: ``ViTForImageClassification(ViTConfig(**model_cfg, num_labels=n_classes))``.
    """
    from transformers import ViTConfig, ViTForImageClassification

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    model_cfg = dict(model_cfg or {})
    name = model_cfg.pop("name", GOOGLE_NAME)
    pretrained = model_cfg.pop("pretrained", mode == "google")
    if mode == "google" and pretrained:
        if int(model_cfg.get("image_size", 224)) != 224:
            raise ValueError("pretrained google ViT requires model.image_size=224")
        return ViTForImageClassification.from_pretrained(name, num_labels=n_classes)
    return ViTForImageClassification(ViTConfig(num_labels=n_classes, **model_cfg))


def make_preprocessor(mode: str, model_cfg: dict | None = None):
    """Batch preprocessing. google: what the notebook's ViTFeatureExtractor(do_rescale=False) did, i.e.
    bilinear resize to the processor size (a no-op at 224) then (x - image_mean) / image_std, with mean/std
    read from the pretrained preprocessor config (0.5 for non-pretrained smoke runs). Implemented in torch so
    it does not depend on the image-processor backend (PIL backends reject values outside [0, 1]).
    scratch: identity (inputs are already z-scored)."""
    if mode != "google":
        return lambda x: x
    import torch
    import torch.nn.functional as F

    model_cfg = model_cfg or {}
    size = int(model_cfg.get("image_size", 224))
    mean, std = [0.5] * 3, [0.5] * 3
    if model_cfg.get("pretrained", True):
        from transformers import ViTImageProcessor

        proc = ViTImageProcessor.from_pretrained(model_cfg.get("name", GOOGLE_NAME))
        mean, std = proc.image_mean, proc.image_std
        sz = proc.size
        size = int(sz["height"] if isinstance(sz, dict) else getattr(sz, "height", size))

    def preprocess(x):
        if x.shape[-2:] != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        sd = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return (x - m) / sd

    return preprocess


def resolve_device(name: str = "auto"):
    import torch

    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        log.warning("device=cuda requested but CUDA is unavailable; using cpu")
    return torch.device("cpu")


def warmup_cosine_lambda(warmup: int, max_steps: int, eta_min: float = 0.0, base_lr: float = 1.0):
    """LR multiplier equal to the notebook's WarmupCosineAnnealingScheduler (indexed by scheduler step).

    Inside ``max_steps`` this reproduces the notebooks. The notebooks counted epochs as optimiser steps
    (scheduler.step() per batch, warmup/max given in epochs), so with enough batches per epoch they ran
    past ``max_steps``; here the step is clamped and the LR stays at the floor (``eta_min``) afterwards.
    """
    ratio = eta_min / base_lr
    max_steps = max(int(max_steps), 1)
    warmup = min(max(int(warmup), 0), max_steps)  # warmup >= max_steps: ramp only, no division by zero

    def f(step: int) -> float:
        if step >= max_steps:  # clamped: the LR stays at the floor once the schedule is over
            return ratio
        if step < warmup:
            return (step + 1) / warmup
        cos = 0.5 * (1 + math.cos((step - warmup) * math.pi / (max_steps - warmup)))
        return ratio + (1 - ratio) * cos

    return f


class AugmentedDataset:
    """Applies :func:`augment_data` to a dataset with a seeded Generator.

    The Generator is keyed on (seed, epoch, worker id, worker torch seed) so noise differs between workers
    and between epochs (call :meth:`set_epoch` before each epoch; DataLoader workers are re-created per epoch
    and get a fresh torch seed, which is mixed in as well) yet stays reproducible for a given seed.
    """

    def __init__(self, base, seed: int, noise_factor=0.05, scale_factor=0.1):
        self.base, self.seed = base, seed
        self.noise_factor, self.scale_factor = noise_factor, scale_factor
        self.epoch = 0
        self._rng, self._key = None, None

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _worker_context(self):
        """(worker id, worker torch seed) or (None, None) in the main process."""
        import torch

        info = torch.utils.data.get_worker_info()
        if info is None:
            return None, None
        return info.id, torch.initial_seed()

    def _get_rng(self):
        wid, wseed = self._worker_context()
        key = (self.epoch, wid, wseed)
        if self._rng is None or key != self._key:
            entropy = [self.seed, self.epoch]
            if wid is not None:
                entropy += [wid + 1, int(wseed) % 2**32]
            self._rng, self._key = np.random.default_rng(entropy), key
        return self._rng

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return augment_data(x, self._get_rng(), self.noise_factor, self.scale_factor).astype(np.float32), y


def is_better(val_acc, val_loss, best_acc, best_loss) -> bool:
    """Checkpoint selection: higher validation accuracy; ties are broken by strictly lower validation loss."""
    return val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)


def _batch_stats(logits, targets, loss):
    return float(loss.item()), int((logits.argmax(1) == targets).sum().item()), int(targets.size(0))


def predict(model, loader, device, preprocess):
    """Return (y_true, y_pred) numpy arrays over ``loader``."""
    import torch

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(pixel_values=preprocess(x.to(device))).logits
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


def fit(model, train_loader, val_loader, tcfg: dict, device, preprocess, best_path, tracker=None):
    """Train; save the best-validation-accuracy state_dict to ``best_path``. Returns (history, best_epoch).

    The test set is never seen here.
    """
    import torch

    tracker = tracker or Tracker()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                                  weight_decay=float(tcfg.get("weight_decay", 0.01)),
                                  eps=float(tcfg.get("eps", 1e-8)))
    scfg = tcfg.get("scheduler")
    scheduler = None
    if scfg:
        if scfg.get("name") != "warmup_cosine":
            raise ValueError("scheduler.name must be 'warmup_cosine'")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda(
            int(scfg["warmup_epochs"]), int(scfg["max_epochs"]), float(scfg.get("eta_min", 0.0)),
            float(tcfg["lr"])))
    clip = tcfg.get("grad_clip")
    model.to(device)
    # DataParallel only when several GPUs are present; state_dicts are always saved from the bare module.
    net = torch.nn.DataParallel(model) if device.type == "cuda" and torch.cuda.device_count() > 1 else model
    history, best_val, best_loss, best_epoch = [], -1.0, math.inf, -1
    for epoch in range(int(tcfg["epochs"])):
        if hasattr(getattr(train_loader, "dataset", None), "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        net.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = net(pixel_values=preprocess(x)).logits
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # per batch, as in the scratch notebook
            lo, c, n = _batch_stats(logits, y, loss)
            loss_sum, correct, total = loss_sum + lo, correct + c, total + n
        train_loss, train_acc = loss_sum / len(train_loader), correct / total

        net.eval()
        vloss_sum, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = net(pixel_values=preprocess(x)).logits
                lo, c, n = _batch_stats(logits, y, criterion(logits, y))  # loss of THIS batch
                vloss_sum, vcorrect, vtotal = vloss_sum + lo, vcorrect + c, vtotal + n
        val_loss, val_acc = vloss_sum / len(val_loader), vcorrect / vtotal

        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        log.info("epoch %d/%s train %.4f/%.4f val %.4f/%.4f", epoch + 1, tcfg["epochs"], train_loss, train_acc,
                 val_loss, val_acc)
        if is_better(val_acc, val_loss, best_val, best_loss):
            best_val, best_loss, best_epoch = val_acc, val_loss, epoch + 1
            torch.save(model.state_dict(), best_path)
    return history, best_epoch


def run_vit(cfg: dict, mode: str, out_dir="runs", tracker: Tracker | None = None):
    """Full pipeline for a resolved config; writes and returns the run directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader

    from .cli import _unique_run_dir, resolve_split

    cfg = copy.deepcopy(cfg)
    seed = int(cfg["seed"])
    set_seed(seed, bool(cfg.get("deterministic", False)))
    task, label_kind = cfg["task"], cfg["label"]
    forced_reason = resolve_split(cfg, task)
    if not float(cfg.get("val_size", 0)) > 0:
        raise ValueError("val_size must be > 0: the checkpoint is selected on a validation split")

    files = list_mat_files(cfg["data_dir"])
    if not files:
        raise FileNotFoundError(f"no .mat files under {cfg['data_dir']!r}")
    parse = extract_label_from_filename if label_kind == "gesture" else extract_group_from_filename
    raw = [parse(f) for f in files]
    if any(v is None for v in raw):
        raise ValueError(f"could not parse a {label_kind} label from every filename")
    labels, classes = remap_labels(raw)
    n_classes = len(classes)
    if cfg.get("n_classes") not in (None, n_classes):
        log.warning("config n_classes=%s but %d distinct labels found in %s", cfg["n_classes"], n_classes,
                    cfg["data_dir"])

    train_idx, val_idx, test_idx, split_info = split_files_info(
        files, labels, mode=cfg["split"], test_size=cfg["test_size"], val_size=cfg["val_size"], seed=seed,
        stratify=cfg.get("stratify", True))
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("validation or test split is empty; add data or adjust test_size/val_size")

    def pick(idx):
        return [files[i] for i in idx], labels[idx]

    (train_f, train_y), (val_f, val_y), (test_f, test_y) = pick(train_idx), pick(val_idx), pick(test_idx)

    model_cfg = dict(cfg.get("model") or {})
    size = int(model_cfg.get("image_size", 224))
    image_shape = (3, size, size)
    tl = int(cfg["target_length"])
    stats = channel_stats(train_f, tl, image_shape)  # per-channel, train files only

    def make_ds(f, y):
        return EMGDataset(f, y, tl, norm_stats=stats, image_shape=image_shape)

    train_ds = make_ds(train_f, train_y)
    acfg = cfg.get("augment") or {}
    if acfg.get("enabled"):
        train_ds = AugmentedDataset(train_ds, seed, acfg.get("noise_factor", 0.05), acfg.get("scale_factor", 0.1))
    tcfg = cfg["train"]
    bs, nw = int(tcfg["batch_size"]), int(tcfg.get("num_workers", 0))
    kw = {"num_workers": nw, "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=data_generator(seed), **kw)
    val_loader = DataLoader(make_ds(val_f, val_y), batch_size=bs, shuffle=False, **kw)
    test_loader = DataLoader(make_ds(test_f, test_y), batch_size=bs, shuffle=False, **kw)

    device = resolve_device(cfg.get("device", "auto"))
    model = build_vit(mode, n_classes, model_cfg)
    preprocess = make_preprocessor(mode, model_cfg)

    method = f"vit-{mode}"
    run_dir = _unique_run_dir(out_dir, task, method, seed)
    with open(run_dir / "config.resolved.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    tracker = tracker or Tracker(cfg, experiment=task)
    tracker.start(run_name=run_dir.name, params=cfg)
    try:
        best_path = run_dir / "best_model.pth"
        history, best_epoch = fit(model, train_loader, val_loader, tcfg, device, preprocess, best_path, tracker)
        with open(run_dir / "history.csv", "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)

        # Single test evaluation, after training, with the validation-selected checkpoint.
        model.load_state_dict(torch.load(best_path, map_location=device))
        y_true, y_pred = predict(model, test_loader, device, preprocess)
        res = report_eval(y_true, y_pred, labels=np.arange(n_classes), names=classes,
                          title=f"Test set (ViT {mode}, {task})")

        metrics = {
            "task": task, "method": method, "split": cfg["split"], "split_forced_reason": forced_reason,
            **split_report(labels, train_idx, val_idx, test_idx, split_info), "seed": seed, "device": device.type,
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "n_classes": int(n_classes), "epochs": len(history), "best_epoch": best_epoch,
            "best_val_accuracy": history[best_epoch - 1]["val_acc"],
            "test_accuracy": res["accuracy"], "macro_f1": res["macro_f1"], "chance_uniform": res["chance_uniform"],
            "chance_majority": res["chance_majority"],
        }
        with open(run_dir / "metrics.json", "w", encoding="utf8") as f:
            json.dump(metrics, f, indent=2)
        res["fig"].savefig(run_dir / "confusion.png", dpi=120)
        plt.close(res["fig"])
        with open(run_dir / "per_class.csv", "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["label", "support", "accuracy"])
            for lab, sup, acc in zip(classes, res["support"], res["per_class_accuracy"]):
                w.writerow([int(lab), int(sup), "" if np.isnan(acc) else float(acc)])
        tracker.log_metrics({"test_accuracy": res["accuracy"], "best_val_accuracy": metrics["best_val_accuracy"]})
        for name in ("config.resolved.yaml", "metrics.json", "history.csv", "per_class.csv", "confusion.png"):
            tracker.log_artifact(run_dir / name)
    finally:
        tracker.end()
    log.info("test accuracy %.4f, wrote %s", res["accuracy"], run_dir)
    return run_dir
