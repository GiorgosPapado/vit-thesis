"""File listing, filename parsing, splits, normalisation statistics and dataset.

Filename scheme (from the original notebooks): the basename is split on ``_`` and the
token containing ``gesture`` (or ``participant``) gives the label (or group), e.g.
``session1_participant7_gesture10_trial1.mat`` -> gesture 10, participant 7.
.mat files hold the EMG window under the key ``data_emg``, shape (time, channels).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import scipy.io as sio
from sklearn.model_selection import GroupShuffleSplit, train_test_split

MAT_KEY = "data_emg"
VIT_IMAGE_SHAPE = (3, 224, 224)


def _extract_int(filepath: str, token: str):
    for part in os.path.basename(filepath).split("_"):
        if token in part:
            return int(part.replace(token, "").split(".")[0])
    return None


def extract_label_from_filename(filepath: str):
    """Gesture label from the filename, or None if absent."""
    return _extract_int(filepath, "gesture")


def extract_group_from_filename(filepath: str):
    """Participant id from the filename, or None if absent."""
    return _extract_int(filepath, "participant")


def list_mat_files(root: str) -> list:
    """Sorted list of ``*.mat`` files directly under ``root``."""
    return sorted(glob.glob(os.path.join(root, "*.mat")))


def load_sample(filepath: str, target_length: int, image_shape=None) -> np.ndarray:
    """Load one file, zero NaN/inf, pad/truncate time to ``target_length``.

    Returns (channels, time). If ``image_shape`` (e.g. VIT_IMAGE_SHAPE) is given, the
    flattened (time, channels) window is padded/truncated and reshaped to it (ViT input).
    """
    data = np.nan_to_num(sio.loadmat(filepath)[MAT_KEY], nan=0.0, posinf=0.0, neginf=0.0)
    if data.shape[0] < target_length:
        data = np.pad(data, ((0, target_length - data.shape[0]), (0, 0)), constant_values=0)
    else:
        data = data[:target_length]
    if image_shape is None:
        return data.T
    flat = data.flatten()
    n = int(np.prod(image_shape))
    flat = np.pad(flat, (0, n - flat.size)) if flat.size < n else flat[:n]
    return flat.reshape(image_shape)


def channel_stats(files, target_length: int, image_shape=None, eps: float = 1e-8):
    """Per-channel mean/std (axis 0 of ``load_sample`` output). Use TRAIN files only.

    Channels with std < eps get std 1 (zero-std guard).
    """
    n = 0
    s = ss = None
    for f in files:
        x = load_sample(f, target_length, image_shape).astype(np.float64)
        x = x.reshape(x.shape[0], -1)
        if s is None:
            s = np.zeros(x.shape[0])
            ss = np.zeros(x.shape[0])
        s += x.sum(axis=1)
        ss += (x**2).sum(axis=1)
        n += x.shape[1]
    if n == 0:
        raise ValueError("no files given")
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 0.0))
    return mean, np.where(std < eps, 1.0, std)


def split_files(files, labels=None, mode: str = "grouped", test_size: float = 0.2,
                val_size: float = 0.2, seed: int = 42):
    """Split into (train, val, test) index arrays into ``files``.

    mode 'random': per-file train_test_split. mode 'grouped': GroupShuffleSplit on the
    participant id, so splits are participant-disjoint. Validation is carved out of the
    train portion (``val_size`` is a fraction of train+val). ``val_size=0`` gives an
    empty validation set.
    """
    files = list(files)
    idx = np.arange(len(files))
    if mode == "random":
        trainval, test = train_test_split(idx, test_size=test_size, random_state=seed)
        if val_size > 0:
            train, val = train_test_split(trainval, test_size=val_size, random_state=seed)
        else:
            train, val = trainval, idx[:0]
        return train, val, test
    if mode != "grouped":
        raise ValueError("mode must be 'grouped' or 'random'")
    groups = np.array([extract_group_from_filename(f) for f in files], dtype=object)
    if any(g is None for g in groups):
        raise ValueError("could not parse a participant id from every filename")
    y = None if labels is None else np.asarray(labels)
    trainval, test = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, y, groups))
    if val_size > 0:
        rel_tr, rel_va = next(GroupShuffleSplit(1, test_size=val_size, random_state=seed).split(
            trainval, None if y is None else y[trainval], groups[trainval]))
        train, val = trainval[rel_tr], trainval[rel_va]
    else:
        train, val = trainval, idx[:0]
    gt, gv, ge = (set(groups[i]) for i in (train, val, test))
    if (gt & gv) or (gt & ge) or (gv & ge):
        raise AssertionError("participant leakage between splits")
    return train, val, test


class EMGDataset:
    """Map-style dataset (works with ``torch.utils.data.DataLoader``; torch not required).

    ``norm_stats`` = (mean, std) from :func:`channel_stats` on the training files.
    """

    def __init__(self, files, labels, target_length: int, norm_stats=None, image_shape=None):
        if len(files) != len(labels):
            raise ValueError("files and labels differ in length")
        self.files = list(files)
        self.labels = labels
        self.target_length = target_length
        self.norm_stats = norm_stats
        self.image_shape = image_shape

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        x = load_sample(self.files[idx], self.target_length, self.image_shape)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            shape = (-1,) + (1,) * (x.ndim - 1)
            x = (x - np.reshape(mean, shape)) / np.reshape(std, shape)
        return x.astype(np.float32), self.labels[idx]
