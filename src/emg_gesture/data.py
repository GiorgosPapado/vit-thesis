"""File listing, filename parsing, splits, normalisation statistics and dataset.

Filename scheme (from the original notebooks): the basename is split on ``_`` and the
token containing ``gesture`` (or ``participant``) gives the label (or group), e.g.
``session1_participant7_gesture10_trial1.mat`` -> gesture 10, participant 7.
.mat files hold the EMG window under the key ``data_emg``, shape (time, channels).

Two layouts coexist: the notebooks use one flat folder of ``*.mat`` per task (e.g.
``s1_s2_s3_matfiles_10gest/``) with labels parsed from filenames, and so does this package
(``list_mat_files``); the labels themselves are raw ids (gesture 1..10, participant 1..17), not
0..K-1 -- use :func:`remap_labels` where a model needs contiguous class indices (ViT).
"""

from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np
import scipy.io as sio
from sklearn.model_selection import GroupShuffleSplit, train_test_split

log = logging.getLogger("emg_gesture")

MAT_KEY = "data_emg"
VIT_IMAGE_SHAPE = (3, 224, 224)


def _extract_int(filepath: str, token: str):
    """Integer after ``token`` in a ``_``-separated name part, matched strictly (``^token<digits>(.ext)?$``)."""
    pattern = re.compile(rf"^{re.escape(token)}(\d+)(?:\.\w+)?$")
    for part in os.path.basename(filepath).split("_"):
        m = pattern.match(part)
        if m:
            return int(m.group(1))
    return None


def extract_label_from_filename(filepath: str):
    """Gesture label from the filename, or None if absent."""
    return _extract_int(filepath, "gesture")


def extract_group_from_filename(filepath: str):
    """Participant id from the filename, or None if absent."""
    return _extract_int(filepath, "participant")


def extract_session_from_filename(filepath: str):
    """Session number from the ``session<k>`` filename token, or None if absent."""
    return _extract_int(filepath, "session")


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


def image_input_report(filepath: str, target_length: int, image_shape=VIT_IMAGE_SHAPE) -> dict:
    """How much of a (target_length, channels) window fits into ``image_shape`` (see :func:`load_sample`).

    ``load_sample`` flattens the window row-major (time-major) and keeps only the first
    ``prod(image_shape)`` values, so a window longer than that is truncated in time. Computed from the
    channel count of ``filepath``; ``retained_fraction`` is capped at 1.0.
    """
    n_channels = int(sio.loadmat(filepath)[MAT_KEY].shape[1])
    n_values = target_length * n_channels
    capacity = int(np.prod(image_shape))
    kept_time = min(target_length, capacity // n_channels)
    return {"window_shape_time_channels": [int(target_length), n_channels], "image_shape": list(image_shape),
            "window_values": int(n_values), "image_values": capacity,
            "retained_fraction": float(min(1.0, capacity / n_values)), "retained_time_samples": int(kept_time),
            "truncated": bool(n_values > capacity), "padded": bool(n_values < capacity),
            "svm_uses_full_window": True}


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


def load_flattened(filepath: str, target_length: int) -> np.ndarray:
    """SVM input: the padded/truncated (time, channels) window flattened row-major."""
    return load_sample(filepath, target_length).T.flatten()


def remap_labels(labels, classes=None):
    """Map arbitrary labels (e.g. gestures 3, 7, 10) to contiguous ints 0..K-1.

    ``classes`` is the sorted array of original labels; computed from ``labels`` when None
    (pass the result of a first call to map val/test consistently). Returns
    ``(mapped, classes)``; ``classes[k]`` is the original label of index ``k``.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels) if classes is None else np.asarray(classes)
    if len(classes) == 0:
        raise ValueError("no classes")
    idx = np.clip(np.searchsorted(classes, labels), 0, len(classes) - 1)
    if not np.array_equal(classes[idx], labels):
        raise ValueError("labels contain values not in classes")
    return idx.astype(np.int64), classes


def _stratified_random(idx, labels, test_size, val_size, seed):
    trainval, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels[idx])
    if val_size > 0:
        train, val = train_test_split(trainval, test_size=val_size, random_state=seed,
                                      stratify=labels[trainval])
    else:
        train, val = trainval, idx[:0]
    return train, val, test


def _session_split(files, y, val_size, seed, train_sessions, test_sessions, info, val_sessions=None):
    """Cross-session split (thesis protocol): train on ``train_sessions``, test on ``test_sessions``.

    Validation is a random per-file fraction (``val_size``) of the TRAIN-session files; the thesis
    described no validation set, so this is our choice. Files from other sessions are excluded.

    The random per-file validation split puts sibling repetitions of the same recording session in train and
    validation, so validation accuracy is optimistic and checkpoint selection is weak (a warning is logged).
    ``val_sessions`` (a non-empty proper subset of ``train_sessions``) instead holds out whole sessions for
    validation, ignoring ``val_size``.
    """
    train_sessions = sorted({int(k) for k in train_sessions})
    test_sessions = sorted({int(k) for k in test_sessions})
    if not train_sessions or not test_sessions:
        raise ValueError("train_sessions and test_sessions must both be non-empty")
    if set(train_sessions) & set(test_sessions):
        raise ValueError("train_sessions and test_sessions must be disjoint")
    sess = [extract_session_from_filename(f) for f in files]
    bad = [os.path.basename(f) for f, k in zip(files, sess) if k is None]
    if bad:
        raise ValueError(f"session split: no 'session<k>' token in {len(bad)} filename(s), e.g. {bad[0]!r}")
    sess = np.array(sess)
    trainval = np.flatnonzero(np.isin(sess, train_sessions))
    test = np.flatnonzero(np.isin(sess, test_sessions))
    if len(trainval) == 0 or len(test) == 0:
        raise ValueError(f"session split: no files for train sessions {train_sessions} or test sessions "
                         f"{test_sessions} (found sessions {sorted(set(sess.tolist()))})")
    info["val_split"] = "none"
    if val_sessions:
        val_sessions = sorted({int(k) for k in val_sessions})
        if not set(val_sessions) < set(train_sessions):
            raise ValueError("val_sessions must be a non-empty proper subset of train_sessions")
        is_val = np.isin(sess[trainval], val_sessions)
        train, val = trainval[~is_val], trainval[is_val]
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"session split: no files for train/validation after holding out {val_sessions}")
        info["val_split"], info["val_sessions"] = "session_holdout", val_sessions
    elif val_size > 0:
        info["val_split"] = "random_per_file"
        log.warning("Session split: validation is a random per-file fraction of sessions %s, so sibling "
                    "repetitions are in both train and validation; validation accuracy is optimistic and "
                    "checkpoint selection is weak. Set val_sessions to hold out a whole session instead.",
                    train_sessions)
        strat = None
        if y is not None:
            _, counts = np.unique(y[trainval], return_counts=True)
            if counts.min() >= 2:
                strat = y[trainval]
        try:
            train, val = train_test_split(trainval, test_size=val_size, random_state=seed, stratify=strat)
            info["stratified"] = strat is not None
        except ValueError:
            train, val = train_test_split(trainval, test_size=val_size, random_state=seed)
    else:
        train, val = trainval, trainval[:0]
    if (set(train) & set(val)) or (set(train) & set(test)) or (set(val) & set(test)):
        raise AssertionError("a file appears in two splits")
    info["train_sessions"], info["test_sessions"] = train_sessions, test_sessions
    if y is not None:
        allc = set(np.unique(y).tolist())
        for name, sub in (("train", train), ("val", val), ("test", test)):
            missing = sorted(allc - set(y[sub].tolist())) if len(sub) else []
            if missing:
                info["missing_classes"][name] = [int(m) for m in missing]
                log.warning("Session split: class(es) %s absent from the %s split.", missing, name)
    return train, val, test


def split_files_info(files, labels=None, mode: str = "grouped", test_size: float = 0.2,
                     val_size: float = 0.2, seed: int = 42, stratify=None,
                     train_sessions=(1, 2), test_sessions=(3,), val_sessions=None):
    """Split into (train, val, test) index arrays plus an info dict.

    mode 'random': per-file train_test_split, stratified by label unless ``stratify`` is False
    (None/True = try; falls back to unstratified with a warning when a class has < 2 members, no labels
    are given, or sklearn cannot stratify the requested sizes). mode 'grouped': GroupShuffleSplit on the
    participant id, so splits are participant-disjoint; a warning is logged when a class is absent from a
    non-empty split. Validation is carved out of the train portion (``val_size`` is a fraction of
    train+val); ``val_size=0`` gives an empty validation set.

    mode 'session': cross-session split (see :func:`_session_split`); ``test_size`` is ignored and
    ``val_size`` is a fraction of the train-session files.

    ``info`` has ``mode``, ``stratified`` (bool), ``stratify_fallback_reason`` (str or None) and, in grouped
    mode, ``missing_classes`` ({'train'|'val'|'test': [labels absent from that non-empty split]}).
    """
    files = list(files)
    idx = np.arange(len(files))
    y = None if labels is None else np.asarray(labels)
    info = {"mode": mode, "stratified": False, "stratify_fallback_reason": None, "missing_classes": {},
            "train_sessions": None, "test_sessions": None, "val_split": None, "val_sessions": None}
    if mode == "session":
        return (*_session_split(files, y, val_size, seed, train_sessions, test_sessions, info,
                                              val_sessions), info)
    if mode == "random":
        out = None
        if stratify is not False:
            reason = None
            if y is None:
                reason = "no labels given"
            else:
                _, counts = np.unique(y, return_counts=True)
                if counts.min() < 2:
                    reason = "a class has fewer than 2 members"
            if reason is None:
                try:
                    out = _stratified_random(idx, y, test_size, val_size, seed)
                    info["stratified"] = True
                except ValueError as exc:
                    reason = f"sklearn could not stratify ({exc})"
            if out is None:
                info["stratify_fallback_reason"] = reason
                log.warning("Stratified random split not possible (%s); using an unstratified split.", reason)
        if out is None:
            trainval, test = train_test_split(idx, test_size=test_size, random_state=seed)
            if val_size > 0:
                train, val = train_test_split(trainval, test_size=val_size, random_state=seed)
            else:
                train, val = trainval, idx[:0]
            out = (train, val, test)
        return (*out, info)
    if mode != "grouped":
        raise ValueError("mode must be 'grouped', 'random' or 'session'")
    groups = np.array([extract_group_from_filename(f) for f in files], dtype=object)
    if any(g is None for g in groups):
        raise ValueError("could not parse a participant id from every filename")
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
    if y is not None:
        allc = set(np.unique(y).tolist())
        for name, sub in (("train", train), ("val", val), ("test", test)):
            if len(sub) == 0:
                continue
            missing = sorted(allc - set(y[sub].tolist()))
            if missing:
                info["missing_classes"][name] = [int(m) for m in missing]
                log.warning("Grouped split: class(es) %s absent from the %s split (few participants); "
                            "metrics on it are not comparable across classes.", missing, name)
    return train, val, test, info


def split_files(files, labels=None, mode: str = "grouped", test_size: float = 0.2,
                val_size: float = 0.2, seed: int = 42, stratify=None,
                train_sessions=(1, 2), test_sessions=(3,), val_sessions=None):
    """:func:`split_files_info` without the info dict: returns (train, val, test) index arrays."""
    return split_files_info(files, labels, mode, test_size, val_size, seed, stratify,
                            train_sessions, test_sessions, val_sessions)[:3]


def split_report(labels, train, val, test, info: dict) -> dict:
    """Metrics fields describing a split: mode, stratification, missing classes, per-split class counts."""
    labels = np.asarray(labels)

    def counts(sub):
        u, c = np.unique(labels[sub], return_counts=True)
        return {str(int(k)): int(v) for k, v in zip(u, c)}

    return {
        "split_mode": info["mode"],
        # Only the grouped split keeps participants disjoint; random has no group key and the session split
        # reuses the same participants in train and test (within-subject, cross-session).
        "generalises_to_new_subjects": info["mode"] == "grouped",
        "val_split": info.get("val_split"),
        "val_sessions": info.get("val_sessions"),
        "train_sessions": info.get("train_sessions"),
        "test_sessions": info.get("test_sessions"),
        "split_stratified": bool(info["stratified"]),
        "split_stratify_fallback_reason": info["stratify_fallback_reason"],
        "split_missing_classes": info["missing_classes"],
        "class_counts": {"train": counts(train), "val": counts(val), "test": counts(test)},
    }


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
