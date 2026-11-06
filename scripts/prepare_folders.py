"""Build the per-task dataset folders (port of ``03_experiments/archive/folders_creation.ipynb``).

Files are COPIED (never moved) from a GrabMyo-style source tree into
``<dest-dir>/<task folder>``. The selection logic follows the notebook cell for each task:

==============  ==========================================================  =======================
task            selection (notebook cell)                                   dest sub-folder
==============  ==========================================================  =======================
10gest          cell 7: ``<src>/session1_participant*/*.dat``, gestures     s1_s2_s3_matfiles_10gest
                {1,2,5,8,9,10,11,12,14,16}, all participants
5gest           cell 9: ``<src>/*.mat`` (flat), gestures {5,8,10,11,16}     s1_s2_s3_matfiles_5gest
17participants   cell 10: ``<src>/*.mat`` (flat), the 10-gesture set above,  17participants
                participant id <= 17
10part          cell 6: ``<src>/session3_participant*`` (flat), participant s1_s2_s3_matfiles_10part
                id 1..10, all gestures
5part           no active cell (see quirks): cell 6 logic with id 1..5      s1_s2_s3_matfiles_5part
==============  ==========================================================  =======================

Open quirks of the notebook, kept on purpose (selection is NOT silently "fixed"):

* 10gest reads ``.dat`` files from per-participant folders while the other tasks read flat ``.mat``
  files; the ``emg-gesture`` CLI only reads ``*.mat``, so the notebook's ``dataset_10gest`` output
  is not directly what ``--task 10gest`` expects. The original ``s1_s2_s3_matfiles_10gest`` folder
  (all sessions, ``.mat``) is not produced by any notebook cell. Point ``--source-dir`` at a tree
  of ``session1_participant*`` folders holding the files you want.
* 5part has no cell of its own; the only trace is a commented-out ``participant_id <= 5`` line in
  cells 9 and 10. It is implemented as the 10part cell with ids 1..5 (an inference).
* The notebook's other cells (0-5, 8) are not reproduced: cell 0 does ``sorted(<path string>)`` and so
  iterates over characters (copies nothing useful); cell 5 filters gestures 1..5 on flat files
  (folder ``VIT_5gestures_20participants``); cells 1-4 and 8 are older Session1/Session3 ``.dat``
  variants with gestures 1..10 / all / 1..5+7..16 (folders ``10gestures_20participants``,
  ``allgestures_10participants``, ``allgestures_20participants``, ``complete_dataset``).
* In 17participants a ``.mat`` whose name has no ``participant<id>`` token and a selected gesture makes
  the notebook crash with ``TypeError`` (``None <= 17``); here it raises ``ValueError``.
* 10part/5part: the notebook calls ``shutil.copy`` on every match of ``session3_participant*``; here
  directories are skipped. Ids are parsed from the ``_``-separated filename token containing
  ``gesture`` / ``participant`` (a non-numeric remainder raises ``ValueError``, as ``int()`` did).
* Only one session prefix is hard-wired per task (``session1_`` for 10gest, ``session3_`` for
  10part/5part), exactly as in the notebook; flat tasks 5gest/17participants accept any ``*.mat``.
* The notebook's 5gest/17participants gesture test is not guarded against files without a gesture
  token (they are simply not selected); the same here.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

GESTURES_10 = frozenset({1, 2, 5, 8, 9, 10, 11, 12, 14, 16})
GESTURES_5 = frozenset({5, 8, 10, 11, 16})


@dataclass(frozen=True)
class TaskSpec:
    folder: str
    pattern: str  # glob relative to the source dir (flat) or to each participant dir
    participant_dirs: str | None = None  # glob of participant folders, or None for a flat source
    gestures: frozenset[int] | None = None
    max_participant: int | None = None


TASKS: dict[str, TaskSpec] = {
    "10gest": TaskSpec("s1_s2_s3_matfiles_10gest", "*.dat", "session1_participant*", GESTURES_10),
    "5gest": TaskSpec("s1_s2_s3_matfiles_5gest", "*.mat", None, GESTURES_5),
    "17participants": TaskSpec("17participants", "*.mat", None, GESTURES_10, 17),
    "10part": TaskSpec("s1_s2_s3_matfiles_10part", "session3_participant*", None, None, 10),
    "5part": TaskSpec("s1_s2_s3_matfiles_5part", "session3_participant*", None, None, 5),
}


def _token_int(path: Path, key: str) -> int | None:
    for part in path.name.split("_"):
        if key in part:
            return int(part.replace(key, ""))
    return None


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", p.name)]


def select_files(source: Path, task: str) -> list[Path]:
    """Return the files the notebook cell for ``task`` would copy (sorted, deterministic)."""
    spec = TASKS[task]
    if spec.participant_dirs:
        dirs = sorted((d for d in source.glob(spec.participant_dirs) if d.is_dir()), key=_natural_key)
        candidates = [f for d in dirs for f in sorted(d.glob(spec.pattern))]
    else:
        candidates = sorted(source.glob(spec.pattern))
    out = []
    for f in candidates:
        if not f.is_file():
            continue
        if spec.gestures is not None and _token_int(f, "gesture") not in spec.gestures:
            continue
        if spec.max_participant is not None:
            pid = _token_int(f, "participant")
            if pid is None:
                raise ValueError(f"no participant id in filename: {f.name}")
            if not 1 <= pid <= spec.max_participant:
                continue
        out.append(f)
    return out


def prepare(source: Path, dest_root: Path, task: str, dry_run: bool = False, force: bool = False) -> int:
    """Copy the selected files for ``task``; return the number of files (copied or planned)."""
    dest = dest_root / TASKS[task].folder
    if dest.exists() and any(dest.iterdir()) and not force and not dry_run:
        raise FileExistsError(f"{dest} is not empty; use --force to copy into it anyway")
    files = select_files(source, task)
    print(f"[{task}] {len(files)} file(s) -> {dest}" + (" (dry run)" if dry_run else ""))
    if dry_run:
        return len(files)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copy the files of each thesis task into its own folder.")
    ap.add_argument("--source-dir", type=Path, required=True, help="root of the raw dataset copy")
    ap.add_argument("--dest-dir", type=Path, required=True, help="where the task folders are created")
    ap.add_argument("--task", default="all", choices=["all", *TASKS])
    ap.add_argument("--dry-run", action="store_true", help="only print what would be copied")
    ap.add_argument("--force", action="store_true", help="copy into a non-empty destination")
    args = ap.parse_args(argv)
    if not args.source_dir.is_dir():
        ap.error(f"--source-dir {args.source_dir} is not a directory")
    tasks = list(TASKS) if args.task == "all" else [args.task]
    try:
        for t in tasks:
            prepare(args.source_dir, args.dest_dir, t, args.dry_run, args.force)
    except (FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
