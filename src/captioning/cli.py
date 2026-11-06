"""``captioning`` command: subset | remove-empty | train | infer."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .data import copy_subset, remove_empty_files
from .paths import resolve_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="captioning")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("subset", help="copy the first N files of a folder")
    s.add_argument("--data-dir", default=None, help="source root (env: DATA_DIR)")
    s.add_argument("--dest-dir", default=None, help="destination root (env: DEST_DIR)")
    s.add_argument("--subdir", default="val2014_desc")
    s.add_argument("-n", type=int, default=5000)
    r = sub.add_parser("remove-empty", help="list (default) or delete zero-byte files")
    r.add_argument("--data-dir", default=None, help="dataset root (env: DATA_DIR)")
    r.add_argument("--subdir", default="train2014_desc")
    r.add_argument("--delete", action="store_true", help="actually delete (default is dry run)")
    sub.add_parser("train", help="train (args forwarded to captioning.train)", add_help=False)
    sub.add_parser("infer", help="caption images (args forwarded to captioning.infer)", add_help=False)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, rest = build_parser().parse_known_args(argv)
    if args.command == "train":
        from . import train

        return train.main(rest)
    if args.command == "infer":
        from . import infer

        return infer.main(rest)
    if rest:
        raise SystemExit(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "subset":
        src = resolve_dir(args.data_dir, "DATA_DIR") / args.subdir
        dst = resolve_dir(args.dest_dir, "DEST_DIR") / args.subdir
        print(f"copied {len(copy_subset(src, dst, args.n))} files to {dst}")
    else:
        folder = resolve_dir(args.data_dir, "DATA_DIR") / args.subdir
        empties = remove_empty_files(folder, dry_run=not args.delete)
        print(f"{'deleted' if args.delete else 'found'} {len(empties)} empty files in {folder}")


if __name__ == "__main__":
    main()
