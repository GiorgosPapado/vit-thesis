"""Synthetic .mat fixtures mirroring the real files: key ``data_emg``, shape (time, channels)."""

import numpy as np
import pytest
import scipy.io as sio

N_CHANNELS = 4


def make_mat_files(root, participants=(1, 2, 3, 4, 5, 6), gestures=(1, 2, 3), trials=2,
                   n_time=50, n_channels=N_CHANNELS, seed=0):
    """Write tiny files named like session1_participant3_gesture2_trial1.mat."""
    rng = np.random.default_rng(seed)
    paths = []
    for p in participants:
        for g in gestures:
            for t in range(1, trials + 1):
                path = root / f"session1_participant{p}_gesture{g}_trial{t}.mat"
                sio.savemat(str(path), {"data_emg": rng.normal(size=(n_time + t, n_channels))})
                paths.append(str(path))
    return paths


@pytest.fixture
def mat_dir(tmp_path):
    make_mat_files(tmp_path)
    return tmp_path


def pytest_configure(config):
    """With REQUIRE_TORCH=1 (make test-all) a missing torch is an error, not a silent skip."""
    import importlib.util
    import os

    if os.environ.get("REQUIRE_TORCH") == "1":
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            raise pytest.UsageError(f"REQUIRE_TORCH=1 but not installed: {', '.join(missing)}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """With REQUIRE_TORCH=1, any skipped test fails the run (skips would hide torch tests)."""
    import os

    if os.environ.get("REQUIRE_TORCH") == "1" and terminalreporter.stats.get("skipped"):
        terminalreporter.write_line("REQUIRE_TORCH=1: tests were skipped, failing", red=True)
        terminalreporter._session.exitstatus = 1
