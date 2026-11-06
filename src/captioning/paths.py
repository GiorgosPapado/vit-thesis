"""Directory resolution from CLI args or environment (no hard-coded machine paths)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_dir(value: Optional[str], env_var: str, *, must_exist: bool = False) -> Path:
    """Return ``value`` if given, else ``$env_var``; raise ValueError when neither is set."""
    chosen = value or os.environ.get(env_var)
    if not chosen:
        raise ValueError(f"no directory given: pass an argument or set the {env_var} environment variable")
    path = Path(chosen).expanduser()
    if must_exist and not path.is_dir():
        raise FileNotFoundError(f"{env_var} directory does not exist: {path}")
    return path
