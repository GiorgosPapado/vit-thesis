import json
from pathlib import Path

import numpy as np
import pytest

from emg_gesture.cli import main
from emg_gesture.features import EMGFeatureExtractor, features_pipeline

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_hand_computed_single_channel():
    x = np.array([[1.0, -2.0, 3.0, 0.5, -1.0]])
    out = EMGFeatureExtractor(n_channels=1).transform(x)
    mean = x.mean()
    expected = [
        np.sqrt((1 + 4 + 9 + 0.25 + 1) / 5),   # rms
        (1 + 2 + 3 + 0.5 + 1) / 5,             # mav
        3 + 5 + 2.5 + 1.5,                     # wl
        3,                                     # zc: crossings 1->-2, -2->3, 0.5->-1 (3->0.5 stays positive)
        2,                                     # ssc: at -2 (down,up) and at 3 (up,down); 0.5: down,down (monotone)
        ((x - mean) ** 2).mean(),              # var
    ]
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out[0], expected)


def test_multi_channel_layout_and_order():
    a = np.array([1.0, 2.0, 4.0])      # channel 0
    b = np.array([-1.0, 1.0, -1.0])    # channel 1
    flat = np.stack([a, b], axis=1).reshape(1, -1)  # (time, channels) row-major
    out = EMGFeatureExtractor(n_channels=2, features=("mav", "wl", "zc")).transform(flat)
    np.testing.assert_allclose(out[0], [7 / 3, 3.0, 0, 1.0, 4.0, 2])
    names = EMGFeatureExtractor(n_channels=2, features=("mav", "wl", "zc")).get_feature_names_out()
    assert list(names) == ["ch0_mav", "ch0_wl", "ch0_zc", "ch1_mav", "ch1_wl", "ch1_zc"]


def test_output_shape_and_unknown_feature():
    X = np.random.default_rng(0).normal(size=(5, 40 * 4))
    assert EMGFeatureExtractor(n_channels=4).fit_transform(X).shape == (5, 24)
    assert EMGFeatureExtractor(n_channels=4, features=("rms", "mav")).transform(X).shape == (5, 8)
    with pytest.raises(ValueError):
        EMGFeatureExtractor(n_channels=4, features=("nope",)).transform(X)


def test_unknown_feature_fails_at_construction():
    from emg_gesture.runs import load_config

    cfg = load_config(CONFIGS, "features", "10gest", [])
    cfg["features"]["features"] = ["rms", "nope"]
    with pytest.raises(ValueError, match="nope"):
        features_pipeline(cfg, 4)


def test_pipeline_classifier_choice():
    from emg_gesture.runs import load_config

    cfg = load_config(CONFIGS, "features", "10gest", ["seed=5"])
    assert type(features_pipeline(cfg, 4).named_steps["clf"]).__name__ == "LinearDiscriminantAnalysis"
    cfg["features"]["classifier"] = "rf"
    assert features_pipeline(cfg, 4).named_steps["clf"].random_state == 5
    cfg["features"]["classifier"] = "svm"
    assert type(features_pipeline(cfg, 4).named_steps["clf"]).__name__ == "SVC"
    cfg["features"]["classifier"] = "bogus"
    with pytest.raises(ValueError):
        features_pipeline(cfg, 4)


def _argv(mat_dir, out, *extra):
    return ["features", "--task", "10gest", "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}", "--set", "target_length=40", *extra]


@pytest.mark.parametrize("clf", ["lda", "rf"])
def test_features_cli_end_to_end(clf, mat_dir, tmp_path):
    out = tmp_path / "runs"
    assert main(_argv(mat_dir, out, "--seed", "3", "--set", f"features.classifier={clf}",
                      "--set", "features.rf.n_estimators=10")) == 0
    (run,) = [p for p in out.iterdir() if p.name.startswith("10gest_features_3_")]
    for name in ("config.resolved.yaml", "metrics.json", "confusion.png", "per_class.csv"):
        assert (run / name).stat().st_size > 0
    m = json.loads((run / "metrics.json").read_text())
    assert m["classifier"] == clf and m["feature_names"] == ["rms", "mav", "wl", "zc", "ssc", "var"]
    assert 0.0 <= m["test_accuracy"] <= 1.0
    assert not [k for k in m if k.startswith("svm_")]


def test_features_seeds_aggregate(mat_dir, tmp_path):
    out = tmp_path / "runs"
    assert main(_argv(mat_dir, out, "--seeds", "1,2")) == 0
    (agg,) = [p for p in out.iterdir() if p.name.startswith("10gest_features-aggregate_")]
    data = json.loads((agg / "aggregate.json").read_text())
    assert data["method"] == "features" and data["seeds"] == [1, 2] and data["n_runs"] == 2
