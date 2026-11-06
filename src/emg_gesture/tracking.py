"""Optional MLflow tracking. A no-op unless enabled in the config AND mlflow is importable.

The tracking URI comes from ``tracking.uri`` in the config or the ``MLFLOW_TRACKING_URI`` environment
variable (config wins). Absolute local paths are rejected so no personal paths end up in configs or
run metadata; use a relative path (``mlruns``), ``sqlite:///...`` or an http(s) server URL.
"""

from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("emg_gesture")

_ABS_PATH = re.compile(r"^([A-Za-z]:[/\\]|[/\\]|file:)")


def check_uri(uri):
    """Return ``uri`` or raise if it is an absolute local path / file URI."""
    if uri and _ABS_PATH.match(str(uri)):
        raise ValueError(f"tracking URI must not be an absolute local path: {uri!r}")
    return uri


def _flatten(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _flatten(v, key + ".")
        else:
            yield key, v


class Tracker:
    """Thin wrapper; every method is safe to call when tracking is inactive."""

    def __init__(self, cfg: dict | None = None, experiment: str | None = None, mlflow_module=None):
        tcfg = (cfg or {}).get("tracking") or {}
        self.mlflow = None
        self.active = False
        if not tcfg.get("enabled", False):
            return
        mlflow = mlflow_module
        if mlflow is None:
            try:
                import mlflow
            except ImportError:
                log.warning("tracking.enabled is true but mlflow is not installed; tracking disabled")
                return
        uri = check_uri(tcfg.get("uri") or os.environ.get("MLFLOW_TRACKING_URI"))
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(tcfg.get("experiment") or experiment or "emg_gesture")
        self.mlflow = mlflow
        self.active = True

    def start(self, run_name=None, params=None):
        if self.active:
            self.mlflow.start_run(run_name=run_name)
            if params:
                self.mlflow.log_params({k: str(v)[:250] for k, v in _flatten(params)})

    def log_metrics(self, metrics: dict, step=None):
        if self.active:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path):
        if self.active:
            self.mlflow.log_artifact(os.fspath(path))

    def end(self):
        if self.active:
            self.mlflow.end_run()
