import json
from pathlib import Path

import numpy as np
import sklearn.model_selection as ms

from emg_gesture.cli import main

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--set", "target_length=40", "--set", "svm.pca_components=5"]
SMALL_GRID = ["--set", "svm.grid.C=[1,10]", "--set", "svm.grid.kernel=[linear,rbf]",
              "--set", "svm.grid.gamma=[scale]"]


def _run(mat_dir, out, *extra):
    argv = ["svm", "--task", "10gest", "--config-dir", str(ROOT / "configs"), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}", *FAST, *extra]
    assert main(argv) == 0
    (run,) = [p for p in Path(out).iterdir() if p.name.startswith("10gest_svm_")]
    return json.loads((run / "metrics.json").read_text())


def test_default_is_fixed(mat_dir, tmp_path):
    m = _run(mat_dir, tmp_path / "r")
    assert m["svm_hyperparameters"] == "fixed (no grid search)"
    assert "best_params" not in m and len(m["cv_scores"]) == 3
    assert m["cv_is_selection_biased"] is False


def test_grid_search_metrics(mat_dir, tmp_path):
    m = _run(mat_dir, tmp_path / "r", "--set", "svm.grid_search=true", *SMALL_GRID)
    assert m["svm_hyperparameters"] == "grid_search"
    assert m["grid"] == {"C": [1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale"]}
    bp = m["best_params"]
    assert bp["C"] in (1, 10) and bp["kernel"] in ("linear", "rbf")
    assert 0.0 <= m["best_cv_score"] <= 1.0
    assert m["cv_is_selection_biased"] is True
    assert "cv_scores" not in m and "cv_mean" not in m
    assert 0.0 <= m["test_accuracy"] <= 1.0


def test_param_grid_has_no_linear_gamma_duplicates():
    from emg_gesture.cli import _svm_param_grid

    g = _svm_param_grid({"C": [1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]})
    assert g == [{"svc__C": [1, 10], "svc__kernel": ["linear"]},
                 {"svc__C": [1, 10], "svc__kernel": ["rbf"], "svc__gamma": ["scale", "auto"]}]


def test_grid_search_fits_on_train_only_and_picks_argmax(mat_dir, tmp_path, monkeypatch):
    import emg_gesture.cli as cli
    from emg_gesture.data import load_flattened

    seen = {}
    orig_fit = ms.GridSearchCV.fit
    orig_split = cli.split_files_info

    def spy_fit(self, X, y=None, **kw):
        seen["X"] = np.array(X)
        seen["groups"] = kw.get("groups")
        res = orig_fit(self, X, y, **kw)
        seen["gs"] = self
        return res

    def spy_split(files, *a, **kw):
        out = orig_split(files, *a, **kw)
        seen["files"], seen["train"], seen["test"] = files, list(out[0]), list(out[2])
        return out

    monkeypatch.setattr(ms.GridSearchCV, "fit", spy_fit)
    monkeypatch.setattr(cli, "split_files_info", spy_split)
    m = _run(mat_dir, tmp_path / "r", "--set", "svm.grid_search=true", *SMALL_GRID)

    files, tl = seen["files"], 40
    train_rows = np.array([load_flattened(files[i], tl) for i in seen["train"]])
    test_rows = [load_flattened(files[i], tl) for i in seen["test"]]
    assert not set(seen["train"]) & set(seen["test"]) and seen["test"]
    np.testing.assert_array_equal(seen["X"], train_rows)  # exactly the train files, in order
    for row in test_rows:
        assert not any(np.array_equal(row, x) for x in seen["X"])
    assert len(seen["groups"]) == m["n_train"]

    r = seen["gs"].cv_results_
    best = int(np.argmax(r["mean_test_score"]))
    assert r["rank_test_score"][best] == 1
    bp = {k.removeprefix("svc__"): v for k, v in r["params"][best].items()}
    assert m["best_params"] == bp
    assert m["best_cv_score"] == float(r["mean_test_score"][best])


def test_grid_run_records_only_pca_and_grid_and_accepts_scalars(mat_dir, tmp_path):
    m = _run(mat_dir, tmp_path / "r", "--set", "svm.grid_search=true", "--set", "svm.grid.C=1",
             "--set", "svm.grid.kernel=[linear]")
    assert m["grid"]["C"] == [1] and m["best_params"]["C"] == 1
    assert set(m["svm_config"]) == {"pca_components", "grid"}
    assert m["svm_config"]["pca_components"] == 5
