"""One-pass CLI-to-artifacts smoke test across methods (svm, features, vit-scratch x input_repr)."""

import json
from pathlib import Path

import pytest
import yaml

from emg_gesture.cli import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
COMMON = ["target_length=40"]
TINY_VIT = ["model.hidden_size=16", "model.num_hidden_layers=1", "model.num_attention_heads=2",
            "model.intermediate_size=32", "model.image_size=16", "model.patch_size=8",
            "model.pretrained=false", "model.spectrogram.nperseg=8", "train.epochs=1",
            "train.batch_size=4", "device=cpu"]
BASE_ARTIFACTS = ("config.resolved.yaml", "metrics.json", "confusion.png", "per_class.csv")
VIT_ARTIFACTS = ("history.csv", "best_model.pth")

CASES = [
    pytest.param("svm", ["svm.pca_components=5"], None, id="svm"),
    pytest.param("features", [], None, id="features"),
    pytest.param("vit-scratch", TINY_VIT + ["model.input_repr=flatten"], "flatten", id="vit-flatten"),
    pytest.param("vit-scratch", TINY_VIT + ["model.input_repr=spectrogram"], "spectrogram",
                 id="vit-spectrogram"),
]


@pytest.mark.parametrize(("command", "overrides", "input_repr"), CASES)
def test_cli_to_artifacts(command, overrides, input_repr, mat_dir, tmp_path):
    if command.startswith("vit"):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
    out = tmp_path / "runs"
    argv = [command, "--task", "10gest", "--config-dir", str(CONFIGS), "--out-dir", str(out),
            "--set", f"data_dir={mat_dir}"]
    for item in (*COMMON, *overrides):
        argv += ["--set", item]
    assert main(argv) == 0

    (run,) = [p for p in out.iterdir() if p.is_dir()]
    expected = BASE_ARTIFACTS + (VIT_ARTIFACTS if input_repr else ())
    for name in expected:
        assert (run / name).stat().st_size > 0, name
    metrics = json.loads((run / "metrics.json").read_text())
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert yaml.safe_load((run / "config.resolved.yaml").read_text())["task"] == "10gest"
    if input_repr:
        assert metrics["vit_input"]["input_repr"] == input_repr
        if input_repr == "spectrogram":
            assert metrics["vit_input"]["retained_fraction"] is None and metrics["vit_input"]["time_covered"] == 1.0
