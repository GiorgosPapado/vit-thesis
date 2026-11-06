import numpy as np
import pytest

from emg_gesture import extract_group_from_filename, extract_label_from_filename, list_mat_files, split_files


def _setup(mat_dir):
    files = list_mat_files(str(mat_dir))
    return files, [extract_label_from_filename(f) for f in files]


def test_grouped_is_participant_disjoint(mat_dir):
    files, labels = _setup(mat_dir)
    tr, va, te = split_files(files, labels, mode="grouped", test_size=0.34, val_size=0.25, seed=42)
    g = [{extract_group_from_filename(files[i]) for i in s} for s in (tr, va, te)]
    assert all(g)
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(files)))


def test_grouped_deterministic(mat_dir):
    files, labels = _setup(mat_dir)
    a = split_files(files, labels, seed=7)
    b = split_files(files, labels, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_random_partition(mat_dir):
    files, labels = _setup(mat_dir)
    tr, va, te = split_files(files, labels, mode="random", test_size=0.25, val_size=0.2)
    assert len(te) == 9 and len(set(tr) | set(va) | set(te)) == len(files)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te))


def test_no_val(mat_dir):
    files, labels = _setup(mat_dir)
    _, va, _ = split_files(files, labels, val_size=0)
    assert len(va) == 0


def test_errors(mat_dir):
    files, _ = _setup(mat_dir)
    with pytest.raises(ValueError):
        split_files(files, mode="bogus")
    with pytest.raises(ValueError):
        split_files(["a_gesture1.mat", "b_gesture2.mat"], mode="grouped")
