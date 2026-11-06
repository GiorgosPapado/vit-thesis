from emg_gesture.runs import deep_merge, resolve_split, unique_run_dir


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}, "k": 1}
    b = {"x": {"y": 5}}
    assert deep_merge(a, b) == {"x": {"y": 5, "z": 2}, "k": 1}
    assert a == {"x": {"y": 1, "z": 2}, "k": 1}


def test_resolve_split_forces_random_for_participants():
    cfg = {"label": "participant", "split": "grouped"}
    assert "participant" in resolve_split(cfg, "5part") and cfg["split"] == "random"
    cfg = {"label": "gesture", "split": "grouped"}
    assert resolve_split(cfg, "10gest") is None and cfg["split"] == "grouped"


def test_unique_run_dir_never_collides(tmp_path):
    a, b = unique_run_dir(tmp_path, "t", "m", 1), unique_run_dir(tmp_path, "t", "m", 1)
    assert a != b and a.is_dir() and b.is_dir()
