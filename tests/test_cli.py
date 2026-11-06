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
        assert (cfg["label"] == "participant") == ("part" in task)
        if cfg["label"] == "participant":
            assert cfg["split"] == "random"


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
    assert metrics["split"] == ("random" if "part" in task else "grouped")


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


def test_parity_with_svm_10gest_notebook(mat_dir, tmp_path):
    """Script pipeline accuracy == accuracy of the code in the svm_10gest notebook.

    Approach: read ``03_experiments/archive/svm_10gest.ipynb`` (not modified), take its code cells for
    loading, splitting and the Pipeline/CV/fit (skipping the seeding and plotting cells), and exec
    them on the synthetic fixture. Only the data folder, target_length (10240 -> 40) and PCA size
    (100 -> 5) are textually patched so the tiny fixture works. The notebook's test accuracy
    (pipe.predict on its test split) must equal ``test_accuracy`` from ``emg-gesture svm`` with
    the same seed, split mode, test size and hyper-parameters, and the CV scores must match too.
    """
    nb_path = ROOT / "03_experiments" / "archive" / "svm_10gest.ipynb"
    assert nb_path.exists(), f"parity notebook missing: {nb_path}"
    cells = [c for c in json.loads(nb_path.read_text(encoding="utf8"))["cells"] if c["cell_type"] == "code"]
    sources = ["".join(c["source"]) for c in cells]
    markers = ("root_folder", "SPLIT_MODE", "ChannelStandardizer")
    wanted = [s for s in sources if any(m in s for m in markers)]
    for m in markers:  # each required piece must be present in some selected cell
        assert any(m in s for s in wanted), f"no notebook cell contains {m!r}"
    ns = {"SEED": 42, "__name__": "nb"}
    for src in wanted:
        src = (src.replace("'s1_s2_s3_matfiles_10gest'", repr(str(mat_dir)))
                  .replace("target_length = 10240", "target_length = 40")
                  .replace("n_components=100", "n_components=5"))
        exec(compile(src, "svm_10gest.ipynb", "exec"), ns)  # noqa: S102
    nb_acc = float(np.mean(ns["pipe"].predict(ns["X_test"]) == ns["y_test"]))
    nb_cv = ns["cross_val_scores"]

    run = _run("10gest", mat_dir, tmp_path / "runs", "--seed", "42")
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["test_accuracy"] == pytest.approx(nb_acc)
    assert metrics["cv_scores"] == pytest.approx(list(nb_cv))
    assert metrics["n_test"] == len(ns["y_test"])


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
