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
