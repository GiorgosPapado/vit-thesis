"""Convert GRABMyo WFDB records (``.hea`` + ``.dat``) to ``.mat`` files the CLI can read.

Each record ``<name>.hea`` / ``<name>.dat`` becomes ``<name>.mat`` holding the single key ``data_emg``
with shape (time, channels) (physical units, float64). The base name is kept, so the
``session<k>`` / ``participant<k>`` / ``gesture<k>`` / ``trial<k>`` tokens survive. The relative
sub-folder layout of ``--source-dir`` is mirrored under ``--dest-dir``, unless ``--flatten`` is given:
then every ``.mat`` is written directly into ``--dest-dir`` (the flat layout ``scripts/prepare_folders.py``
expects) and two records with the same base name are an error.

Checks: ``--channels`` indices must lie in [0, n_sig) and be unique; all records must have the same
channel count; NaN counts per record produce a warning; existing ``.mat`` files are not overwritten
without ``--force``. A ``manifest.json`` (fs, units, sig_name, channels, n_records) is written to the
destination.

Only records with a ``.hea`` header are converted; every other file (``.dat`` without header, ``.txt``,
``RECORDS``, ...) is skipped. Requires ``wfdb`` (``pip install -e ".[convert]"``), imported lazily.

NOT verified on real GRABMyo files (only on synthetic records written with ``wfdb.wrsamp``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _import_wfdb():
    try:
        import wfdb
    except ImportError as e:
        raise SystemExit(
            "error: the 'wfdb' package is required for conversion; install it with "
            "`pip install wfdb` or `pip install -e \".[convert]\"`"
        ) from e
    return wfdb


def find_records(source: Path) -> list[Path]:
    """Return record paths (header path without suffix) that have both ``.hea`` and ``.dat``."""
    out = []
    for hea in sorted(source.rglob("*.hea")):
        if hea.with_suffix(".dat").is_file():
            out.append(hea.with_suffix(""))
    return out


def _plan(records: list[Path], source: Path, dest: Path, flatten: bool) -> list[tuple[Path, Path]]:
    pairs = []
    seen: dict[Path, Path] = {}
    for rec in records:
        out = dest / (rec.name if flatten else str(rec.relative_to(source)))
        out = out.with_name(out.name + ".mat")  # keep dots in base names
        if out in seen:
            raise ValueError(f"base-name collision: {rec} and {seen[out]} would both write {out}")
        seen[out] = rec
        pairs.append((rec, out))
    return pairs


def _uniq(values: list):
    u = []
    for v in values:
        if v not in u:
            u.append(v)
    return u[0] if len(u) == 1 else u


def convert(source: Path, dest: Path, channels: list[int] | None = None, dry_run: bool = False,
            flatten: bool = False, force: bool = False) -> int:
    """Convert all records; return the number converted (or planned for a dry run)."""
    import numpy as np
    import scipy.io as sio

    records = find_records(source)
    pairs = _plan(records, source, dest, flatten)
    if channels is not None:
        if len(set(channels)) != len(channels):
            raise ValueError(f"--channels has duplicate indices: {channels}")
        if any(c < 0 for c in channels):
            raise ValueError(f"--channels must be non-negative: {channels}")
    if dry_run:
        for rec, out in pairs:
            print(f"{rec} -> {out} (dry run)")
        print(f"{len(pairs)} record(s) planned (dry run)")
        return len(pairs)
    wfdb = _import_wfdb()
    if not force:
        existing = [out for _, out in pairs if out.exists()]
        if existing:
            raise ValueError(f"{len(existing)} output file(s) exist, e.g. {existing[0]}; use --force to overwrite")
    headers = {rec: wfdb.rdheader(str(rec)) for rec, _ in pairs}
    counts = {h.n_sig for h in headers.values()}
    if len(counts) > 1:
        raise ValueError(f"records have different channel counts: {sorted(counts)}")
    n_sig = counts.pop() if counts else 0
    if channels is not None and any(c >= n_sig for c in channels):
        raise ValueError(f"--channels {channels} outside [0, {n_sig})")
    idx = list(range(n_sig)) if channels is None else channels
    fs, units, names = [], [], []
    for rec, out in pairs:
        print(f"{rec} -> {out}")
        r = wfdb.rdrecord(str(rec))
        sig = r.p_signal  # (time, channels), physical units
        if sig is None:
            raise ValueError(f"record {rec} has no physical signal")
        sig = np.asarray(sig[:, idx], dtype=np.float64)
        n_nan = int(np.isnan(sig).sum())
        if n_nan:
            print(f"warning: {rec} contains {n_nan} NaN value(s)", file=sys.stderr)
        fs.append(r.fs)
        units.append([r.units[i] for i in idx])
        names.append([r.sig_name[i] for i in idx])
        out.parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(str(out), {"data_emg": sig})
    if pairs:
        manifest = {"fs": _uniq(fs), "units": _uniq(units), "sig_name": _uniq(names), "channels": idx,
                    "n_records": len(pairs), "flatten": flatten}
        dest.mkdir(parents=True, exist_ok=True)
        with open(dest / "manifest.json", "w", encoding="utf8") as f:
            json.dump(manifest, f, indent=2)
    print(f"{len(pairs)} record(s) converted")
    return len(pairs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert GRABMyo WFDB (.dat/.hea) records to .mat (key data_emg).")
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--dest-dir", type=Path, required=True)
    ap.add_argument("--channels", type=str, default=None, help="comma-separated 0-based channel indices to keep")
    ap.add_argument("--flatten", action="store_true", help="write all .mat into --dest-dir (no sub-folders)")
    ap.add_argument("--force", action="store_true", help="overwrite existing .mat files")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be converted")
    args = ap.parse_args(argv)
    if not args.source_dir.is_dir():
        ap.error(f"--source-dir {args.source_dir} is not a directory")
    channels = None
    if args.channels:
        try:
            channels = [int(c) for c in args.channels.split(",") if c.strip()]
        except ValueError:
            ap.error("--channels must be a comma-separated list of integers")
    try:
        convert(args.source_dir, args.dest_dir, channels, args.dry_run, args.flatten, args.force)
    except (ValueError, IndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
