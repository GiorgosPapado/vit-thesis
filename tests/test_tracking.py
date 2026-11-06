import builtins

import pytest

from emg_gesture.tracking import Tracker, check_uri


class FakeMlflow:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


def test_disabled_is_noop():
    fake = FakeMlflow()
    t = Tracker({"tracking": {"enabled": False}}, mlflow_module=fake)
    t.start("r", {"a": 1})
    t.log_metrics({"x": 1.0}, step=0)
    t.log_artifact("nope")
    t.end()
    assert not t.active and fake.calls == []
    assert not Tracker().active and not Tracker({}).active


def test_enabled_without_mlflow_is_noop(monkeypatch):
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mlflow":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    t = Tracker({"tracking": {"enabled": True}})
    t.start("r")
    t.log_metrics({"x": 1})
    t.end()
    assert not t.active


def test_enabled_uses_uri_from_config_then_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    fake = FakeMlflow()
    Tracker({"tracking": {"enabled": True}}, experiment="e", mlflow_module=fake)
    assert ("set_tracking_uri", ("http://env:5000",), {}) in fake.calls
    fake = FakeMlflow()
    Tracker({"tracking": {"enabled": True, "uri": "mlruns"}}, mlflow_module=fake)
    assert ("set_tracking_uri", ("mlruns",), {}) in fake.calls


@pytest.mark.parametrize("uri", ["C:\\Users\\me\\mlruns", "/home/me/mlruns", "file:///home/me/x"])
def test_absolute_paths_rejected(uri):
    with pytest.raises(ValueError):
        check_uri(uri)
    with pytest.raises(ValueError):
        Tracker({"tracking": {"enabled": True, "uri": uri}}, mlflow_module=FakeMlflow())
