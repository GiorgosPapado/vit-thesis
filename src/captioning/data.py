"""Dataset utilities: subset copy, empty-file handling and caption text loading (torch-free)."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]


def copy_subset(src: PathLike, dst: PathLike, n: int = 5000, suffix: str = "") -> List[Path]:
    """Copy the first ``n`` files (sorted by name, optionally filtered by ``suffix``) from src to dst."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {src}")
    files = sorted(p for p in src.iterdir() if p.is_file() and p.name.endswith(suffix))[: max(n, 0)]
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in files:
        target = dst / p.name
        shutil.copy2(p, target)
        copied.append(target)
    return copied


def find_empty_files(folder: PathLike) -> List[Path]:
    """Zero-byte regular files directly inside ``folder`` (sorted)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"directory does not exist: {folder}")
    return sorted(p for p in folder.iterdir() if p.is_file() and p.stat().st_size == 0)


def remove_empty_files(folder: PathLike, dry_run: bool = True) -> List[Path]:
    """Delete zero-byte files in ``folder``; with ``dry_run`` (default) only report them."""
    empties = find_empty_files(folder)
    if not dry_run:
        for p in empties:
            p.unlink()
    return empties


def remove_numbers(lines: List[str]) -> str:
    """Strip digits from each caption line and join them into one text ending with a period."""
    return ". ".join(re.sub(r"\d+", "", line).strip() for line in lines) + "."


def limit_words(text: str, word_limit: int = 200) -> str:
    words = text.split()
    if len(words) > word_limit:
        return " ".join(words[:word_limit]) + "..."
    return text


def load_captions(desc_dir: PathLike, word_limit: int = 50) -> Dict[str, str]:
    """Map ``<image stem>`` -> cleaned caption from ``<stem>_desc.txt`` files in ``desc_dir``."""
    desc_dir = Path(desc_dir)
    if not desc_dir.is_dir():
        raise FileNotFoundError(f"description directory does not exist: {desc_dir}")
    out = {}
    for p in sorted(desc_dir.glob("*_desc.txt")):
        stem = p.name[: -len("_desc.txt")]
        lines = p.read_text(encoding="utf-8").splitlines()
        out[stem] = limit_words(remove_numbers(lines), word_limit)
    return out
