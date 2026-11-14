"""Build the per-task dataset folders.

Files are COPIED (never moved) from a flat folder of ``.mat`` files (the output of
``scripts/convert_dat_to_mat.py --flatten``) into ``<dest-dir>/<task folder>``:

==============  ==========================================================  =======================
task            selection (flat ``<src>/*.mat``)                            dest sub-folder
==============  ==========================================================  =======================
10gest          gestures {1,2,5,8,9,10,11,12,14,16}, all sessions and       s1_s2_s3_matfiles_10gest
                participants
5gest           gestures {5,8,10,11,16}                                     s1_s2_s3_matfiles_5gest
17participants  the 10-gesture set above, participant id <= 17              17participants
10part          ``session3_participant*.mat``, participant id 1..10,        s1_s2_s3_matfiles_10part
                all gestures
5part           ``session3_participant*.mat``, participant id 1..5,         s1_s2_s3_matfiles_5part
                all gestures
==============  ==========================================================  =======================

Rules and quirks, kept on purpose (selection is NOT silently "fixed"):

* All tasks read the flat ``.mat`` files of ``--source-dir`` (no per-participant sub-folders, no ``.dat``).
  10gest keeps every session (the ``--split session`` layout needs sessions 1, 2 and 3).
* 5part has no counterpart in the original thesis code; it is the 10part rule with ids 1..5 (an inference).
* 17participants: a ``.mat`` whose name has no ``participant<id>`` token and a selected gesture raises
  ``ValueError``.
* 10part/5part only use ``session3_`` files, so ``--split session`` cannot work on their folders.
* Ids are parsed from the ``_``-separated filename token containing ``gesture`` / ``participant``
  (a non-numeric remainder raises ``ValueError``). Files without a gesture token are not selected by
  the gesture-filtered tasks. Directories are skipped.
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
    pattern: str  # glob relative to the (flat) source dir
    gestures: frozenset[int] | None = None
    max_participant: int | None = None


TASKS: dict[str, TaskSpec] = {
    "10gest": TaskSpec("s1_s2_s3_matfiles_10gest", "*.mat", GESTURES_10),
    "5gest": TaskSpec("s1_s2_s3_matfiles_5gest", "*.mat", GESTURES_5),
    "17participants": TaskSpec("17participants", "*.mat", GESTURES_10, 17),
    "10part": TaskSpec("s1_s2_s3_matfiles_10part", "session3_participant*.mat", None, 10),
    "5part": TaskSpec("s1_s2_s3_matfiles_5part", "session3_participant*.mat", None, 5),
}


def _token_int(path: Path, key: str) -> int | None:
    for part in path.name.split("_"):
        if key in part:
            return int(part.replace(key, ""))
    return None


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", p.name)]


def select_files(source: Path, task: str) -> list[Path]:
    """Return the files selected for ``task`` (naturally sorted, deterministic)."""
    spec = TASKS[task]
    candidates = sorted(source.glob(spec.pattern), key=_natural_key)
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
