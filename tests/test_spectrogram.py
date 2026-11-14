import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_mat_files
from emg_gesture.data import EMGDataset, channel_stats, image_input_report, load_sample
from emg_gesture.runs import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SIZE = 16
SHAPE = (3, SIZE, SIZE)
TL = 200  # 200 x 4 = 800 values > 3*16*16 = 768: the flatten path truncates


@pytest.fixture
def files(tmp_path):
    return make_mat_files(tmp_path, participants=(1, 2), gestures=(1, 2), trials=1, n_time=TL - 1)


def test_shape_determinism_finite(files):
    a = load_sample(files[0], TL, SHAPE, "spectrogram")
    b = load_sample(files[0], TL, SHAPE, "spectrogram")
    assert a.shape == SHAPE and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, load_sample(files[1], TL, SHAPE, "spectrogram"))


def test_report_not_truncated_where_flatten_is(files):
    flat = image_input_report(files[0], TL, SHAPE)
    spec = image_input_report(files[0], TL, SHAPE, "spectrogram")
    assert flat["truncated"] and flat["input_repr"] == "flatten"
    assert spec["input_repr"] == "spectrogram" and spec["truncated"] is False
    assert spec["retained_fraction"] is None and spec["time_covered"] == 1.0
    assert spec["nperseg"] == 32 and spec["noverlap"] == 16 and spec["n_frequency_bins"] == 17
    assert "identity not preserved" in spec["channel_grouping"]


def test_sinusoid_peaks_at_expected_frequency_row():
    from emg_gesture.data import spectrogram_image

    n, nperseg, k = 256, 32, 4  # bin k <-> k/nperseg cycles/sample
    t = np.arange(n)
    data = np.tile(np.sin(2 * np.pi * (k / nperseg) * t)[:, None], (1, 3))
    size = nperseg // 2 + 1  # one image row per frequency bin (no resampling distortion)
    img = spectrogram_image(data, size, {"nperseg": nperseg, "log": False})
    assert img.shape == (3, size, size)
    for plane in img:
        assert int(np.argmax(plane.mean(axis=1))) == k


def test_stats_and_dataset(files):
    mean, std = channel_stats(files, TL, SHAPE, input_repr="spectrogram")
    assert mean.shape == (3,) and std.shape == (3,) and (std > 0).all()
    ds = EMGDataset(files, [0, 1, 2, 3], TL, norm_stats=(mean, std), image_shape=SHAPE, input_repr="spectrogram")
    x, _ = ds[0]
    assert x.shape == SHAPE and x.dtype == np.float32


def test_flatten_default_unchanged(files):
    np.testing.assert_array_equal(load_sample(files[0], TL, SHAPE), load_sample(files[0], TL, SHAPE, "flatten"))
    with pytest.raises(ValueError):
        load_sample(files[0], TL, SHAPE, "bogus")


def test_default_configs_are_flatten():
    for mode in ("google", "scratch"):
        assert load_config(CONFIGS, f"vit_{mode}", "10gest")["model"]["input_repr"] == "flatten"


def test_scratch_run_spectrogram(mat_dir, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from emg_gesture.cli import main

    out = tmp_path / "runs"
    argv = ["vit-scratch", "--task", "10gest", "--config-dir", str(CONFIGS), "--out-dir", str(out)]
    sets = [f"data_dir={mat_dir}", "model.image_size=16", "model.patch_size=8", "model.hidden_size=16",
            "model.num_attention_heads=2", "model.num_hidden_layers=1", "model.intermediate_size=32",
            "model.pretrained=false", "model.input_repr=spectrogram", "train.epochs=1", "train.batch_size=4",
            "device=cpu", "target_length=40"]
    for s in sets:
        argv += ["--set", s]
    assert main(argv) == 0
    (run,) = [p for p in out.iterdir() if p.name.startswith("10gest_vit-scratch_")]
    vi = json.loads((run / "metrics.json").read_text())["vit_input"]
    assert vi["input_repr"] == "spectrogram" and vi["truncated"] is False
    assert vi["retained_fraction"] is None and vi["time_covered"] == 1.0


def test_burst_in_last_part_of_long_window_changes_image():
    from emg_gesture.data import spectrogram_image

    rng = np.random.default_rng(0)
    base = rng.normal(size=(2000, 3)) * 0.01
    burst = base.copy()
    burst[-200:] += rng.normal(size=(200, 3)) * 5.0  # only in the final 10% of the window
    # 2000 x 3 = 6000 values > 3*16*16 = 768: a truncating (flatten) input would never see the burst
    a, b = spectrogram_image(base, 16), spectrogram_image(burst, 16)
    assert not np.allclose(a, b)
    assert np.abs(a[..., -2:] - b[..., -2:]).max() > 10 * np.abs(a[..., :4] - b[..., :4]).max()


def test_three_group_channel_averaging():
    from emg_gesture.data import spectrogram_image

    rng = np.random.default_rng(1)
    data = rng.normal(size=(128, 6))
    img = spectrogram_image(data, 8, {"log": False})
    singles = [spectrogram_image(np.tile(data[:, [c]], (1, 3)), 8, {"log": False})[0] for c in range(6)]
    for k, grp in enumerate(([0, 1], [2, 3], [4, 5])):
        np.testing.assert_allclose(img[k], np.mean([singles[c] for c in grp], axis=0), atol=1e-10)
