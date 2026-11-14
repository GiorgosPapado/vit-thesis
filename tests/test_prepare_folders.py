"""Tests for scripts/prepare_folders.py on a tiny synthetic tree."""
import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "prepare_folders", Path(__file__).resolve().parents[1] / "scripts" / "prepare_folders.py"
)
pf = importlib.util.module_from_spec(_spec)
sys.modules["prepare_folders"] = pf
_spec.loader.exec_module(pf)


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


@pytest.fixture()
def src(tmp_path):
    root = tmp_path / "src"
    for pid in (1, 2, 11, 18):
        for g in (1, 3, 5, 8, 16):
            _touch(root / f"session1_participant{pid}_gesture{g}_trial1.mat")
            _touch(root / f"session2_participant{pid}_gesture{g}_trial1.mat")
            _touch(root / f"session3_participant{pid}_gesture{g}_trial1.mat")
    (root / "session3_participant99_dir").mkdir()
    return root


def test_10gest_flat_mat_all_sessions(src):
    files = pf.select_files(src, "10gest")
    assert len(files) == 3 * 4 * 4  # sessions 1-3 x participants x gestures 1,5,8,16 (3 is not in the set)
    assert all(f.suffix == ".mat" for f in files)
    assert {f.name.split("_")[0] for f in files} == {"session1", "session2", "session3"}


def test_5gest_flat_mat(src):
    files = pf.select_files(src, "5gest")
    assert {pf._token_int(f, "gesture") for f in files} == {5, 8, 16}
    assert all(f.suffix == ".mat" for f in files)


def test_17participants_cap(src):
    files = pf.select_files(src, "17participants")
    assert {pf._token_int(f, "participant") for f in files} == {1, 2, 11}
    assert {pf._token_int(f, "gesture") for f in files} == {1, 5, 8, 16}


def test_10part_and_5part_all_gestures_skip_dirs(src):
    assert len(pf.select_files(src, "10part")) == 2 * 5  # participants 1, 2
    assert len(pf.select_files(src, "5part")) == 2 * 5


def test_17participants_without_id_raises(src):
    _touch(src / "weird_gesture5.mat")
    with pytest.raises(ValueError):
        pf.select_files(src, "17participants")


def test_copy_not_move_and_refuse_nonempty(src, tmp_path):
    dest = tmp_path / "out"
    args = ["--source-dir", str(src), "--dest-dir", str(dest), "--task", "5gest"]
    assert pf.main(args) == 0
    out = dest / "s1_s2_s3_matfiles_5gest"
    assert len(list(out.iterdir())) == 4 * 3 * 3
    assert (src / "session3_participant1_gesture5_trial1.mat").exists()
    assert pf.main(args) == 1
    assert pf.main([*args, "--force"]) == 0


def test_dry_run_creates_nothing(src, tmp_path):
    dest = tmp_path / "out"
    assert pf.main(["--source-dir", str(src), "--dest-dir", str(dest), "--dry-run"]) == 0
    assert not dest.exists()


def test_all_creates_every_folder(src, tmp_path):
    dest = tmp_path / "out"
    assert pf.main(["--source-dir", str(src), "--dest-dir", str(dest)]) == 0
    assert {p.name for p in dest.iterdir()} == {t.folder for t in pf.TASKS.values()}
