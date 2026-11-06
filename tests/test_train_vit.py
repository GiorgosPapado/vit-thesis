import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from emg_gesture import train_vit  # noqa: E402
from emg_gesture.cli import TASKS, load_config, main  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TINY_MODEL = {"image_size": 32, "patch_size": 8, "hidden_size": 16, "num_attention_heads": 2,
              "num_hidden_layers": 1, "intermediate_size": 32}
TINY = [f"model.{k}={v}" for k, v in TINY_MODEL.items()] + [
    "model.pretrained=false", "train.epochs=3", "train.batch_size=4", "device=cpu", "target_length=40"]


def _run(mode, task, mat_dir, out, *extra):
    argv = [f"vit-{mode}", "--task", task, "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}"]
    for item in (*TINY, *extra):
        argv += ["--set", item]
    assert main(argv) == 0
    (run,) = [p for p in Path(out).iterdir() if p.name.startswith(f"{task}_vit-{mode}_")]
    return run


def test_config_merge_all_tasks():
    for mode, ep, bs in (("google", 50, 32), ("scratch", 500, 256)):
        for task in TASKS:
            cfg = load_config(CONFIGS, f"vit_{mode}", task)
            assert cfg["train"]["epochs"] == ep and cfg["train"]["batch_size"] == bs
            assert cfg["test_size"] == 0.2 and cfg["val_size"] == 0.2 and cfg["train"]["lr"] == 1e-4


@pytest.mark.parametrize("mode", ["google", "scratch"])
def test_smoke_both_modes(mode, mat_dir, tmp_path):
    run = _run(mode, "10gest", mat_dir, tmp_path / "runs", "seed=3")
    for name in ("config.resolved.yaml", "metrics.json", "confusion.png", "per_class.csv", "history.csv",
                 "best_model.pth"):
        assert (run / name).stat().st_size > 0
    with open(run / "history.csv", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and float(rows[0]["val_loss"]) > 0
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["epochs"] == 3 and 0 <= metrics["test_accuracy"] <= 1
    val = [float(r["val_acc"]) for r in rows]
    assert 1 <= metrics["best_epoch"] <= 3 and rows[metrics["best_epoch"] - 1]["val_acc"] == str(max(val))
    assert metrics["macro_f1"] is not None and metrics["split_mode"] == "grouped"
    assert set(metrics["class_counts"]) == {"train", "val", "test"}
    assert metrics["split"] == "grouped"
    assert yaml.safe_load((run / "config.resolved.yaml").read_text())["seed"] == 3


def test_participant_task_forces_random(mat_dir, tmp_path):
    run = _run("scratch", "5part", mat_dir, tmp_path / "runs", "train.epochs=1")
    assert json.loads((run / "metrics.json").read_text())["split"] == "random"


def test_test_eval_once_after_training_with_val_selected_checkpoint(mat_dir, tmp_path, monkeypatch):
    events = []
    real_fit, real_report, real_predict = train_vit.fit, train_vit.report_eval, train_vit.predict

    def fit(*a, **k):
        out = real_fit(*a, **k)
        events.append(("fit_done", out))
        return out

    def predict(model, loader, *a, **k):
        events.append(("predict", len(loader.dataset)))
        return real_predict(model, loader, *a, **k)

    def report(*a, **k):
        events.append(("report",))
        return real_report(*a, **k)

    monkeypatch.setattr(train_vit, "fit", fit)
    monkeypatch.setattr(train_vit, "predict", predict)
    monkeypatch.setattr(train_vit, "report_eval", report)
    run = _run("scratch", "10gest", mat_dir, tmp_path / "runs")
    assert [e[0] for e in events] == ["fit_done", "predict", "report"]  # one test pass, after training
    metrics = json.loads((run / "metrics.json").read_text())
    assert events[1][1] == metrics["n_test"]
    history, best_epoch = events[0][1]
    assert best_epoch == max(range(len(history)),
                             key=lambda i: (history[i]["val_acc"], -history[i]["val_loss"], -i)) + 1
    assert metrics["best_val_accuracy"] == history[best_epoch - 1]["val_acc"]


def test_fit_tracks_best_validation_epoch():
    import torch

    model = train_vit.build_vit("scratch", 3, TINY_MODEL)
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    loader = [(x[:4], y[:4]), (x[4:], y[4:])]
    with tempfile.TemporaryDirectory() as d:
        hist, best = train_vit.fit(model, loader, loader, {"epochs": 2, "lr": 1e-3}, torch.device("cpu"),
                                   lambda t: t, Path(d) / "b.pth")
        assert (Path(d) / "b.pth").exists()
    assert len(hist) == 2 and best in (1, 2)


def test_warmup_cosine_matches_notebook_formula():
    f = train_vit.warmup_cosine_lambda(5, 500)
    assert f(0) == pytest.approx(0.2) and f(4) == pytest.approx(1.0)
    assert f(5) == pytest.approx(1.0) and f(255) == pytest.approx(0.5 * (1 + np.cos(250 * np.pi / 495)))


def test_build_vit_rejects_bad_mode():
    with pytest.raises(ValueError):
        train_vit.build_vit("nope", 3)


def test_warmup_cosine_clamped_past_max_steps():
    f = train_vit.warmup_cosine_lambda(5, 500)
    assert f(499) > 0.0 and f(500) == 0.0 and f(5000) == 0.0  # floor after the schedule ends
    g = train_vit.warmup_cosine_lambda(5, 500, eta_min=1e-6, base_lr=1e-4)
    assert g(500) == pytest.approx(0.01) and g(10**6) == pytest.approx(0.01)
    assert g(255) == pytest.approx(0.01 + 0.99 * 0.5 * (1 + np.cos(250 * np.pi / 495)))  # inside: unchanged


def test_warmup_cosine_warmup_not_below_max_no_zero_division():
    for warmup, mx in ((10, 10), (20, 10), (0, 10)):
        f = train_vit.warmup_cosine_lambda(warmup, mx)
        vals = [f(s) for s in range(0, 30)]
        assert all(0.0 <= v <= 1.0 for v in vals) and vals[-1] == 0.0


def test_is_better_tie_breaks_on_lower_val_loss():
    assert train_vit.is_better(0.5, 1.0, 0.4, 0.1)
    assert not train_vit.is_better(0.4, 0.1, 0.5, 1.0)
    assert train_vit.is_better(0.5, 0.8, 0.5, 0.9)  # tie on accuracy, lower loss wins
    assert not train_vit.is_better(0.5, 0.9, 0.5, 0.8)
    assert not train_vit.is_better(0.5, 0.9, 0.5, 0.9)  # full tie keeps the earlier epoch


def test_fit_keeps_lower_loss_epoch_on_accuracy_tie(monkeypatch):
    import torch

    model = train_vit.build_vit("scratch", 3, TINY_MODEL)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 0])
    saved = []
    monkeypatch.setattr(torch, "save", lambda sd, path: saved.append(path))
    losses = iter([2.0, 1.0, 1.5])
    monkeypatch.setattr(train_vit, "_batch_stats", lambda logits, t, loss: (next(losses) if not loss.requires_grad
                                                                              else 0.0, 2, 4))
    with tempfile.TemporaryDirectory() as d:
        _, best = train_vit.fit(model, [(x, y)], [(x, y)], {"epochs": 3, "lr": 1e-3}, torch.device("cpu"),
                                lambda t: t, Path(d) / "b.pth")
    assert best == 2  # equal val accuracy every epoch; epoch 2 has the lowest val loss


def test_run_vit_channel_stats_uses_train_files_only_and_split_is_participant_disjoint(mat_dir, tmp_path,
                                                                                       monkeypatch):
    from emg_gesture.data import extract_group_from_filename

    seen = {}
    real_stats, real_split = train_vit.channel_stats, train_vit.split_files_info

    def stats(files, *a, **k):
        seen["stats"] = list(files)
        return real_stats(files, *a, **k)

    def split(files, *a, **k):
        out = real_split(files, *a, **k)
        seen["files"], seen["idx"] = list(files), out[:3]
        return out

    monkeypatch.setattr(train_vit, "channel_stats", stats)
    monkeypatch.setattr(train_vit, "split_files_info", split)
    _run("scratch", "10gest", mat_dir, tmp_path / "runs", "train.epochs=1")
    tr, va, te = ([seen["files"][i] for i in idx] for idx in seen["idx"])
    assert seen["stats"] == tr
    assert not set(seen["stats"]) & (set(va) | set(te))
    groups = [{extract_group_from_filename(p) for p in part} for part in (tr, va, te)]
    assert all(groups)
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])


def test_vit_scratch_seeds_aggregate(mat_dir, tmp_path):
    out = tmp_path / "runs"
    argv = ["vit-scratch", "--task", "10gest", "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--seeds", "1,2", "--set", f"data_dir={mat_dir}"]
    for item in (*TINY, "train.epochs=1"):
        argv += ["--set", item]
    assert main(argv) == 0
    (agg,) = [p for p in out.iterdir() if "aggregate" in p.name]
    data = json.loads((agg / "aggregate.json").read_text())
    assert data["seeds"] == [1, 2] and len(data["per_seed"]) == 2 and "macro_f1" in data
