"""YAML config loading with ``key=value`` overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

DEFAULT_CONFIG = Path("configs") / "captioning.yaml"


def load_config(path: Optional[str] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load YAML (default configs/captioning.yaml if it exists) then apply ``key=value`` overrides."""
    cfg: Dict[str, Any] = {}
    p = Path(path) if path else DEFAULT_CONFIG
    if path or p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        cfg[key.strip()] = yaml.safe_load(raw)
    return cfg
