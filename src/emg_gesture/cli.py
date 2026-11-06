"""Command line entry point: ``emg-gesture svm --task 10gest --split grouped --seed 42``."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .data import (
    extract_group_from_filename,
    extract_label_from_filename,
    list_mat_files,
    load_flattened,
    split_files,
)
from .evaluate import report_eval
from .models import svm_pipeline
from .seed import set_seed

log = logging.getLogger("emg_gesture")

TASKS = ("10gest", "5gest", "17participants", "10part", "5part")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply ``key=value`` overrides; dotted keys address nested sections, values are YAML-parsed."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, last = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(raw)
    return cfg


def load_config(config_dir, method: str, task: str, overrides=None) -> dict:
    """Merge base.yaml, <method>.yaml and task/<task>.yaml (later wins), then --set overrides."""
    config_dir = Path(config_dir)
    cfg = {}
    for path in (config_dir / "base.yaml", config_dir / f"{method}.yaml", config_dir / "task" / f"{task}.yaml"):
        cfg = deep_merge(cfg, _read_yaml(path))
    return apply_overrides(cfg, overrides)


def _unique_run_dir(out_dir, task, method, seed) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(out_dir) / f"{task}_{method}_{seed}_{stamp}"
    path, n = base, 1
    while path.exists():
        path = Path(f"{base}_{n}")
        n += 1
    path.mkdir(parents=True)
    return path


def run_svm(cfg: dict, out_dir="runs") -> Path:
    """Run the SVM baseline for a resolved config and write the run directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

    cfg = copy.deepcopy(cfg)
    seed = int(cfg["seed"])
    set_seed(seed)
    task, label_kind = cfg["task"], cfg["label"]
    if label_kind not in ("gesture", "participant"):
        raise ValueError("label must be 'gesture' or 'participant'")
    if label_kind == "participant" and cfg["split"] != "random":
        log.warning("Task %s uses participant as the label: a participant-grouped split would remove every "
                    "class from the training set, so the split is forced to 'random' (was %r).", task, cfg["split"])
        cfg["split"] = "random"
    if cfg["split"] not in ("grouped", "random"):
        raise ValueError("split must be 'grouped' or 'random'")

    files = list_mat_files(cfg["data_dir"])
    if not files:
        raise FileNotFoundError(f"no .mat files under {cfg['data_dir']!r}")
    parse = extract_label_from_filename if label_kind == "gesture" else extract_group_from_filename
    labels = [parse(f) for f in files]
    if any(lab is None for lab in labels):
        raise ValueError(f"could not parse a {label_kind} label from every filename")
    labels = np.array(labels)
    n_found = len(np.unique(labels))
    if cfg.get("n_classes") not in (None, n_found):
        log.warning("config n_classes=%s but %d distinct labels found in %s", cfg["n_classes"], n_found,
                    cfg["data_dir"])

    train_idx, _, test_idx = split_files(files, labels, mode=cfg["split"], test_size=cfg["test_size"],
                                         val_size=0.0, seed=seed)
    X_train = np.array([load_flattened(files[i], cfg["target_length"]) for i in train_idx])
    X_test = np.array([load_flattened(files[i], cfg["target_length"]) for i in test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    n_channels = X_train.shape[1] // cfg["target_length"]

    pipe = svm_pipeline(cfg, n_channels)
    if cfg["split"] == "grouped":
        cv_groups = np.array([extract_group_from_filename(files[i]) for i in train_idx], dtype=object)
        cv = GroupKFold(n_splits=cfg["cv_splits"])
    else:
        cv_groups = None
        cv = StratifiedKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, groups=cv_groups, scoring="accuracy")
    pipe.fit(X_train, y_train)  # test set untouched until here
    y_pred = pipe.predict(X_test)
    classes = np.unique(np.concatenate([y_train, y_test]))
    res = report_eval(y_test, y_pred, labels=classes, title=f"Test set (SVM, {task})")

    run_dir = _unique_run_dir(out_dir, task, "svm", seed)
    with open(run_dir / "config.resolved.yaml", "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metrics = {
        "task": task,
        "split": cfg["split"],
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes_seen": int(len(classes)),
        "test_accuracy": res["accuracy"],
        "chance_uniform": res["chance_uniform"],
        "chance_majority": res["chance_majority"],
        "cv_scores": [float(s) for s in cv_scores],
        "cv_mean": float(np.mean(cv_scores)),
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
    log.info("test accuracy %.4f, wrote %s", res["accuracy"], run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emg-gesture", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("svm", help="train/evaluate the SVM baseline")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--split", choices=("grouped", "random"), default=None, help="default: from config")
    p.add_argument("--seed", type=int, default=None, help="default: from config")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value (dotted keys ok, e.g. svm.kernel=rbf); repeatable")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config_dir, args.command, args.task, args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.split is not None:
        cfg["split"] = args.split
    if cfg.get("data_dir") is None:
        raise SystemExit("data_dir is not set (task config or --set data_dir=...)")
    cfg["data_dir"] = os.fspath(cfg["data_dir"])
    run_svm(cfg, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
