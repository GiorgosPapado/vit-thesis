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


def test_random_split_is_stratified_by_default(mat_dir):
    from emg_gesture import split_files
    from emg_gesture.data import split_files_info

    files, labels = _setup(mat_dir)
    y = np.array(labels)
    tr, va, te, info = split_files_info(files, y, mode="random", test_size=0.25, val_size=0.2, seed=1)
    assert info["stratified"] and info["stratify_fallback_reason"] is None
    for part in (tr, va, te):
        assert set(y[part]) == {1, 2, 3}
    assert len(split_files(files, y, mode="random")) == 3
    *_, off = split_files_info(files, y, mode="random", stratify=False)
    assert not off["stratified"] and off["stratify_fallback_reason"] is None


def test_stratify_falls_back_with_warning(caplog):
    import logging

    from emg_gesture.data import split_files_info

    files = [f"s_participant{i}_gesture{1 if i < 9 else 2}.mat" for i in range(10)]  # class 2 has 1 member
    y = np.array([1] * 9 + [2])
    with caplog.at_level(logging.WARNING, logger="emg_gesture"):
        *_, info = split_files_info(files, y, mode="random", test_size=0.3, val_size=0.2)
    assert not info["stratified"] and "fewer than 2" in info["stratify_fallback_reason"]
    assert "unstratified" in caplog.text


def test_grouped_warns_when_class_missing(caplog):
    import logging

    from emg_gesture.data import split_files_info

    files = [f"s_participant{p}_gesture{g}.mat" for p in range(1, 5) for g in ((1, 2) if p < 4 else (1,))]
    y = np.array([int(f.split("gesture")[1].split(".")[0]) for f in files])
    found = False
    for seed in range(20):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="emg_gesture"):
            *_, info = split_files_info(files, y, mode="grouped", test_size=0.25, val_size=0.34, seed=seed)
        if info["missing_classes"]:
            found = True
            assert "absent from" in caplog.text
            break
    assert found


def test_split_report_counts(mat_dir):
    from emg_gesture.data import split_files_info, split_report

    files, labels = _setup(mat_dir)
    y = np.array(labels)
    tr, va, te, info = split_files_info(files, y, mode="grouped", test_size=0.34, val_size=0.25, seed=42)
    rep = split_report(y, tr, va, te, info)
    assert rep["split_mode"] == "grouped"
    assert sum(sum(c.values()) for c in rep["class_counts"].values()) == len(files)
    assert sum(rep["class_counts"]["test"].values()) == len(te)
