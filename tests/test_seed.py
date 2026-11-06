import os

import pytest

from emg_gesture import set_seed

torch = pytest.importorskip("torch")


def test_deterministic_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    prev = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        set_seed(1)
        assert not torch.are_deterministic_algorithms_enabled()
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        set_seed(1, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"]
    finally:
        torch.use_deterministic_algorithms(prev)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
