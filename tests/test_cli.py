import json
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from emg_gesture.cli import TASKS, main
from emg_gesture.runs import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FAST = ["--set", "target_length=40", "--set", "svm.pca_components=5"]


def _run(task, mat_dir, out, *extra):
    argv = ["svm", "--task", task, "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}", *FAST, *extra]
    assert main(argv) == 0
    (run,) = [p for p in Path(out).iterdir() if p.name.startswith(f"{task}_svm_")]
    return run


def test_config_merge_and_task_files():
    cfg = load_config(CONFIGS, "svm", "5gest", ["svm.C=3", "seed=1"])
    assert cfg["svm"]["kernel"] == "rbf" and cfg["svm"]["C"] == 3 and cfg["svm"]["pca_components"] == 100
    assert cfg["seed"] == 1 and cfg["label"] == "gesture" and cfg["n_classes"] == 5
    assert load_config(CONFIGS, "svm", "10gest")["svm"]["kernel"] == "linear"
    for task in TASKS:
        cfg = load_config(CONFIGS, "svm", task)
        assert cfg["label"] in ("gesture", "participant") and cfg["n_classes"] > 0 and cfg["data_dir"]
        assert cfg["split"] == ("random" if cfg["label"] == "participant" else "grouped")
    p17 = load_config(CONFIGS, "svm", "17participants")
    assert (p17["label"], p17["n_classes"], p17["split"]) == ("gesture", 10, "grouped")


@pytest.mark.parametrize("task", TASKS)
def test_svm_smoke_all_tasks(task, mat_dir, tmp_path):
    run = _run(task, mat_dir, tmp_path / "runs", "--seed", "3")
    assert run.name.startswith(f"{task}_svm_3_")
    for name in ("config.resolved.yaml", "metrics.json", "confusion.png", "per_class.csv"):
        assert (run / name).stat().st_size > 0
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0.0 <= metrics["test_accuracy"] <= 1.0 and len(metrics["cv_scores"]) == 3
    resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())
    assert resolved["seed"] == 3 and resolved["task"] == task
    assert metrics["split"] == (
        "random" if load_config(CONFIGS, "svm", task)["label"] == "participant" else "grouped"
    )


def test_participant_task_forces_random_split(mat_dir, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="emg_gesture"):
        run = _run("10part", mat_dir, tmp_path / "runs", "--split", "grouped")
    assert "forced to 'random'" in caplog.text
    assert json.loads((run / "metrics.json").read_text())["split"] == "random"
    assert yaml.safe_load((run / "config.resolved.yaml").read_text())["split"] == "random"


def test_same_seed_reproducible(mat_dir, tmp_path):
    a = json.loads((_run("10gest", mat_dir, tmp_path / "a") / "metrics.json").read_text())
    b = json.loads((_run("10gest", mat_dir, tmp_path / "b") / "metrics.json").read_text())
    assert a == b


def test_parity_with_sklearn_reference(mat_dir, tmp_path):
    """CLI svm result == an independent reference built directly from sklearn.

    The reference uses the same split (split_files_info), ChannelStandardizer -> StandardScaler -> PCA -> SVC
    with the config hyper-parameters and the same seed, on the synthetic fixture. test_accuracy and
    cv_scores must match.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from emg_gesture.data import (
        extract_group_from_filename,
        extract_label_from_filename,
        list_mat_files,
        load_flattened,
        split_files_info,
    )
    from emg_gesture.models import ChannelStandardizer

    seed, length = 42, 40
    cfg = load_config(CONFIGS, "svm", "10gest", ["target_length=40", "svm.pca_components=5"])
    files = list_mat_files(str(mat_dir))
    labels = np.array([extract_label_from_filename(f) for f in files])
    tr, _, te, _ = split_files_info(files, labels, mode=cfg["split"], test_size=cfg["test_size"], val_size=0.0,
                                    seed=seed, stratify=cfg.get("stratify", True))
    X_tr = np.array([load_flattened(files[i], length) for i in tr])
    X_te = np.array([load_flattened(files[i], length) for i in te])
    s = cfg["svm"]
    pipe = Pipeline([
        ("cn", ChannelStandardizer(n_channels=X_tr.shape[1] // length)),
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=5, random_state=seed)),
        ("svc", SVC(kernel=s["kernel"], C=s["C"], gamma=s["gamma"])),
    ])
    groups = np.array([extract_group_from_filename(files[i]) for i in tr], dtype=object)
    ref_cv = cross_val_score(pipe, X_tr, labels[tr], cv=GroupKFold(n_splits=cfg["cv_splits"]), groups=groups,
                             scoring="accuracy")
    ref_acc = float(np.mean(pipe.fit(X_tr, labels[tr]).predict(X_te) == labels[te]))

    run = _run("10gest", mat_dir, tmp_path / "runs", "--seed", str(seed))
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["test_accuracy"] == pytest.approx(ref_acc)
    assert metrics["cv_scores"] == pytest.approx(list(ref_cv))
    assert metrics["n_test"] == len(te)


def test_metrics_record_split_details(mat_dir, tmp_path):
    m = json.loads((_run("5part", mat_dir, tmp_path / "a", "--split", "grouped") / "metrics.json").read_text())
    assert m["split_mode"] == "random" and m["split_stratified"] is True
    assert "participant" in m["split_forced_reason"]
    assert set(m["class_counts"]) == {"train", "val", "test"} and 0 <= m["macro_f1"] <= 1
    g = json.loads((_run("10gest", mat_dir, tmp_path / "b") / "metrics.json").read_text())
    assert g["split_mode"] == "grouped" and g["split_forced_reason"] is None
    assert sum(g["class_counts"]["test"].values()) == g["n_test"]


def test_svm_seeds_writes_aggregate(mat_dir, tmp_path):
    out = tmp_path / "runs"
    argv = ["svm", "--task", "10gest", "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}", *FAST, "--seeds", "1,2,3"]
    assert main(argv) == 0
    runs = [p for p in out.iterdir() if p.name.startswith("10gest_svm_")]
    assert len(runs) == 3
    (agg,) = [p for p in out.iterdir() if "aggregate" in p.name]
    data = json.loads((agg / "aggregate.json").read_text())
    accs = [json.loads((r / "metrics.json").read_text())["test_accuracy"] for r in runs]
    assert data["seeds"] == [1, 2, 3] and data["n_runs"] == 3
    assert data["test_accuracy"]["mean"] == pytest.approx(np.mean(accs))
    assert data["test_accuracy"]["std"] == pytest.approx(np.std(accs, ddof=1))
    assert "macro_f1" in data and len(data["per_seed"]) == 3


def test_seed_and_seeds_conflict(mat_dir, tmp_path):
    with pytest.raises(SystemExit):
        main(["svm", "--task", "10gest", "--config-dir", str(CONFIGS), "--set", f"data_dir={mat_dir}",
              "--seed", "1", "--seeds", "1,2"])
