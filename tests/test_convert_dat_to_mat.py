"""Tests for scripts/convert_dat_to_mat.py using synthetic WFDB records."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

wfdb = pytest.importorskip("wfdb")

_spec = importlib.util.spec_from_file_location(
    "convert_dat_to_mat", Path(__file__).resolve().parents[1] / "scripts" / "convert_dat_to_mat.py"
)
cv = importlib.util.module_from_spec(_spec)
sys.modules["convert_dat_to_mat"] = cv
_spec.loader.exec_module(cv)

NAME = "session1_participant3_gesture2_trial1"


@pytest.fixture()
def src(tmp_path):
    root = tmp_path / "src"
    d = root / "session1_participant3"
    d.mkdir(parents=True)
    sig = np.random.default_rng(0).normal(size=(100, 4))
    wfdb.wrsamp(NAME, fs=2048, units=["mV"] * 4, sig_name=["a", "b", "c", "d"], p_signal=sig,
                fmt=["16"] * 4, write_dir=str(d))
    (d / "notes.txt").write_text("not a record")
    (d / "orphan.dat").write_bytes(b"x")  # .dat without header: skipped
    return root


def test_convert_roundtrip(src, tmp_path):
    dest = tmp_path / "dest"
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest)]) == 0
    out = dest / "session1_participant3" / f"{NAME}.mat"
    assert sorted(p.name for p in dest.rglob("*") if p.is_file()) == sorted([out.name, "manifest.json"])
    data = sio.loadmat(str(out))["data_emg"]
    ref = wfdb.rdrecord(str(src / "session1_participant3" / NAME)).p_signal
    assert data.shape == (100, 4)
    np.testing.assert_allclose(data, ref)


def test_channels_and_dry_run(src, tmp_path, capsys):
    dest = tmp_path / "dest"
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest), "--dry-run"]) == 0
    assert not dest.exists() and "1 record(s) planned" in capsys.readouterr().out
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest), "--channels", "1,3"]) == 0
    data = sio.loadmat(str(dest / "session1_participant3" / f"{NAME}.mat"))["data_emg"]
    assert data.shape == (100, 2)


def test_bad_channel_index_errors(src, tmp_path):
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(tmp_path / "d"), "--channels", "9"]) == 1


def test_duplicate_channels_and_nan_warning_and_manifest(src, tmp_path, capsys):
    dest = tmp_path / "d"
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest), "--channels", "1,1"]) == 1
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest), "--channels", "-1"]) == 1
    assert cv.main(["--source-dir", str(src), "--dest-dir", str(dest), "--channels", "0,2"]) == 0
    import json

    m = json.loads((dest / "manifest.json").read_text())
    assert m["fs"] == 2048 and m["channels"] == [0, 2] and m["sig_name"] == ["a", "c"] and m["n_records"] == 1
    assert m["units"] == ["mV", "mV"]


def test_refuse_overwrite_without_force(src, tmp_path):
    dest = tmp_path / "d"
    args = ["--source-dir", str(src), "--dest-dir", str(dest)]
    assert cv.main(args) == 0
    assert cv.main(args) == 1
    assert cv.main([*args, "--force"]) == 0


def test_nan_warns(tmp_path, capsys):
    root = tmp_path / "s"
    root.mkdir()
    sig = np.zeros((20, 2))
    sig[3, 1] = np.nan
    wfdb.wrsamp("rec", fs=100, units=["mV"] * 2, sig_name=["a", "b"], p_signal=sig, fmt=["16"] * 2,
                write_dir=str(root))
    assert cv.main(["--source-dir", str(root), "--dest-dir", str(tmp_path / "d")]) == 0
    assert "NaN" in capsys.readouterr().err


def _write(root, name, n_ch=3, sub=None):
    d = root / sub if sub else root
    d.mkdir(parents=True, exist_ok=True)
    sig = np.random.default_rng(1).normal(size=(60, n_ch))
    wfdb.wrsamp(name, fs=2048, units=["mV"] * n_ch, sig_name=[f"c{i}" for i in range(n_ch)], p_signal=sig,
                fmt=["16"] * n_ch, write_dir=str(d))


def test_mixed_channel_counts_error(tmp_path):
    root = tmp_path / "s"
    _write(root, "session1_participant1_gesture1_trial1", 3)
    _write(root, "session1_participant1_gesture2_trial1", 4)
    assert cv.main(["--source-dir", str(root), "--dest-dir", str(tmp_path / "d")]) == 1


def test_flatten_collision_errors(tmp_path):
    root = tmp_path / "s"
    _write(root, "session1_participant1_gesture1_trial1", sub="a")
    _write(root, "session1_participant1_gesture1_trial1", sub="b")
    dest = tmp_path / "d"
    assert cv.main(["--source-dir", str(root), "--dest-dir", str(dest), "--flatten"]) == 1
    assert not dest.exists()
    assert cv.main(["--source-dir", str(root), "--dest-dir", str(dest)]) == 0  # nested mirrors fine


def test_nested_convert_flatten_prepare_folders(tmp_path):
    pf_spec = importlib.util.spec_from_file_location(
        "prepare_folders_it", Path(__file__).resolve().parents[1] / "scripts" / "prepare_folders.py")
    pf = importlib.util.module_from_spec(pf_spec)
    sys.modules["prepare_folders_it"] = pf
    pf_spec.loader.exec_module(pf)

    raw = tmp_path / "raw"
    for s in (1, 2, 3):
        for pid in (1, 2):
            for g in (1, 3, 5):
                _write(raw, f"session{s}_participant{pid}_gesture{g}_trial1", sub=f"Session{s}/participant{pid}")
    flat = tmp_path / "flat"
    assert cv.main(["--source-dir", str(raw), "--dest-dir", str(flat), "--flatten"]) == 0
    assert len(list(flat.glob("*.mat"))) == 18
    data = tmp_path / "data"
    assert pf.main(["--source-dir", str(flat), "--dest-dir", str(data), "--task", "all"]) == 0
    g10 = list((data / "s1_s2_s3_matfiles_10gest").glob("*.mat"))
    assert len(g10) == 3 * 2 * 2  # gestures 1 and 5 (3 not in the set), all sessions
    p10 = list((data / "s1_s2_s3_matfiles_10part").glob("*.mat"))
    assert len(p10) == 2 * 3 and all(f.name.startswith("session3_") for f in p10)
    assert len(list((data / "s1_s2_s3_matfiles_5part").glob("*.mat"))) == 6
