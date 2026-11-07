import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from emg_gesture import extract_session_from_filename, split_files
from emg_gesture.cli import main
from emg_gesture.data import split_files_info, split_report

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _names(sessions=(1, 2, 3), participants=(1, 2), gestures=(1, 2), trials=(1, 2, 3)):
    return [f"session{s}_participant{p}_gesture{g}_trial{t}.mat"
            for s in sessions for p in participants for g in gestures for t in trials]


def _labels(files):
    return np.array([int(f.split("gesture")[1].split("_")[0]) for f in files])


def test_extract_session():
    assert extract_session_from_filename("/x/session2_participant7_gesture10_trial1.mat") == 2
    assert extract_session_from_filename("participant7_gesture10.mat") is None


def test_session_split_partitions_by_session():
    files = _names()
    tr, va, te, info = split_files_info(files, _labels(files), mode="session", val_size=0.25, seed=3)
    sess = np.array([extract_session_from_filename(f) for f in files])
    assert set(sess[tr]) | set(sess[va]) == {1, 2} and set(sess[te]) == {3}
    assert len(va) > 0
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) + len(va) == int((sess <= 2).sum()) and len(te) == int((sess == 3).sum())
    assert info["train_sessions"] == [1, 2] and info["test_sessions"] == [3]


def test_session_split_configurable_and_excludes_other_sessions():
    files = _names()
    tr, va, te = split_files(files, mode="session", val_size=0, train_sessions=[1], test_sessions=[3])
    sess = np.array([extract_session_from_filename(f) for f in files])
    assert set(sess[tr]) == {1} and set(sess[te]) == {3} and len(va) == 0


def test_session_split_errors():
    with pytest.raises(ValueError, match="session<k>"):
        split_files(_names() + ["participant1_gesture1.mat"], mode="session")
    with pytest.raises(ValueError, match="disjoint"):
        split_files(_names(), mode="session", train_sessions=[1, 3], test_sessions=[3])
    with pytest.raises(ValueError, match="no files"):
        split_files(_names(sessions=(1, 2)), mode="session")


def test_split_report_records_sessions():
    files = _names()
    y = _labels(files)
    tr, va, te, info = split_files_info(files, y, mode="session")
    rep = split_report(y, tr, va, te, info)
    assert rep["split_mode"] == "session" and rep["train_sessions"] == [1, 2] and rep["test_sessions"] == [3]
    other = split_files_info(files, y, mode="random")
    assert split_report(y, *other[:3], other[3])["train_sessions"] is None


def test_cli_session_metrics(tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    rng = np.random.default_rng(0)
    for n in _names(participants=(1, 2, 3)):
        sio.savemat(str(data / n), {"data_emg": rng.normal(size=(50, 4))})
    out = tmp_path / "runs"
    assert main(["svm", "--task", "10gest", "--split", "session", "--config-dir", str(CONFIGS),
                 "--out-dir", str(out), "--set", f"data_dir={data}", "--set", "target_length=40",
                 "--set", "svm.pca_components=5"]) == 0
    (run,) = list(out.iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert m["split_mode"] == "session" and m["train_sessions"] == [1, 2] and m["test_sessions"] == [3]
    assert m["n_test"] == 18 and m["n_train"] == 36
    assert m["generalises_to_new_subjects"] is False
    assert m["svm_uses_full_window"] is True and m["svm_hyperparameters"] == "fixed (no grid search)"
    assert m["svm_config"]["kernel"] == "linear"


def _session_dir(tmp_path, sessions=(1, 2, 3)):
    data = tmp_path / "d"
    data.mkdir()
    rng = np.random.default_rng(0)
    for n in _names(sessions=sessions, participants=(1, 2, 3)):
        sio.savemat(str(data / n), {"data_emg": rng.normal(size=(50, 4))})
    return data


def test_cli_session_seeds_aggregate_records_split(tmp_path):
    data, out = _session_dir(tmp_path), tmp_path / "runs"
    assert main(["svm", "--task", "10gest", "--split", "session", "--seeds", "1,2", "--config-dir", str(CONFIGS),
                 "--out-dir", str(out), "--set", f"data_dir={data}", "--set", "target_length=40",
                 "--set", "svm.pca_components=5"]) == 0
    (agg,) = [p for p in out.iterdir() if "aggregate" in p.name]
    a = json.loads((agg / "aggregate.json").read_text())
    assert a["split_mode"] == "session" and a["generalises_to_new_subjects"] is False
    assert a["train_sessions"] == [1, 2] and a["test_sessions"] == [3] and "initialisation" in a["std_note"]


def test_cli_session_split_needs_all_sessions(tmp_path):
    """A folder with only session1 files (what scripts/prepare_folders.py yields for 10gest) cannot be split."""
    data = _session_dir(tmp_path, sessions=(1,))
    with pytest.raises(ValueError, match="no files for train sessions"):
        main(["svm", "--task", "10gest", "--split", "session", "--config-dir", str(CONFIGS),
              "--out-dir", str(tmp_path / "runs"), "--set", f"data_dir={data}", "--set", "target_length=40",
              "--set", "svm.pca_components=5"])


def test_generalises_to_new_subjects_flag():
    files = _names(participants=(1, 2, 3, 4, 5))
    y = _labels(files)
    for mode, expected in (("grouped", True), ("random", False), ("session", False)):
        tr, va, te, info = split_files_info(files, y, mode=mode)
        assert split_report(y, tr, va, te, info)["generalises_to_new_subjects"] is expected


def test_session_val_random_warns_and_holdout_does_not(caplog):
    files = _names()
    y = _labels(files)
    with caplog.at_level("WARNING", logger="emg_gesture"):
        *_, info = split_files_info(files, y, mode="session", val_size=0.25)
    assert info["val_split"] == "random_per_file" and "optimistic" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="emg_gesture"):
        tr, va, te, info = split_files_info(files, y, mode="session", val_sessions=[2])
    assert "optimistic" not in caplog.text and info["val_split"] == "session_holdout"
    sess = np.array([extract_session_from_filename(f) for f in files])
    assert set(sess[tr]) == {1} and set(sess[va]) == {2} and set(sess[te]) == {3}
    assert split_report(y, tr, va, te, info)["val_sessions"] == [2]


def test_session_val_sessions_must_be_proper_subset():
    with pytest.raises(ValueError, match="proper subset"):
        split_files(_names(), mode="session", val_sessions=[1, 2])
    with pytest.raises(ValueError, match="proper subset"):
        split_files(_names(), mode="session", val_sessions=[3])


def test_image_input_report_truncation(tmp_path):
    from emg_gesture.data import image_input_report

    path = tmp_path / "session1_participant1_gesture1_trial1.mat"
    sio.savemat(str(path), {"data_emg": np.zeros((10240, 32))})
    r = image_input_report(str(path), 10240, (3, 224, 224))
    assert r["window_shape_time_channels"] == [10240, 32] and r["truncated"] is True
    assert r["retained_time_samples"] == 3 * 224 * 224 // 32
    assert r["retained_fraction"] == pytest.approx(3 * 224 * 224 / (10240 * 32))
    assert r["svm_uses_full_window"] is True
    small = image_input_report(str(path), 100, (3, 224, 224))
    assert small["truncated"] is False and small["retained_fraction"] == 1.0
