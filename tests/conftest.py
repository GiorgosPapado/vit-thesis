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
