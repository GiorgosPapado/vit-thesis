"""Config loading, split resolution and run-directory helpers shared by the CLI and the trainers."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path

import yaml

log = logging.getLogger("emg_gesture")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply ``key=value`` overrides; dotted keys address nested sections, values are YAML-parsed."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, last = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(raw)
    return cfg


def load_config(config_dir, method: str, task: str, overrides=None) -> dict:
    """Merge base.yaml, <method>.yaml and task/<task>.yaml (later wins), then --set overrides."""
    config_dir = Path(config_dir)
    cfg = {}
    for path in (config_dir / "base.yaml", config_dir / f"{method}.yaml", config_dir / "task" / f"{task}.yaml"):
        cfg = deep_merge(cfg, _read_yaml(path))
    return apply_overrides(cfg, overrides)


def unique_run_dir(out_dir, task, method, seed) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(out_dir) / f"{task}_{method}_{seed}_{stamp}"
    path, n = base, 1
    while path.exists():
        path = Path(f"{base}_{n}")
        n += 1
    path.mkdir(parents=True)
    return path


def resolve_split(cfg: dict, task: str):
    """Validate label/split; force 'random' for participant labels. Mutates ``cfg``, returns the reason or None."""
    label_kind = cfg["label"]
    if label_kind not in ("gesture", "participant"):
        raise ValueError("label must be 'gesture' or 'participant'")
    reason = None
    if label_kind == "participant" and cfg["split"] != "random":
        reason = "participant label: a participant-grouped split would remove every class from training"
        log.warning("Task %s uses participant as the label: split forced to 'random' (was %r).", task, cfg["split"])
        cfg["split"] = "random"
    if cfg["split"] not in ("grouped", "random", "session"):
        raise ValueError("split must be 'grouped', 'random' or 'session'")
    return reason
