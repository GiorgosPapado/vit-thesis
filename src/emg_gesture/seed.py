"""Seeding helpers. torch is optional and imported lazily."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed python, numpy and (if installed) torch. Returns the seed.

    ``deterministic=True`` (config key ``deterministic``, default off) additionally sets
    ``CUBLAS_WORKSPACE_CONFIG`` and ``torch.use_deterministic_algorithms(True, warn_only=True)``. This is
    best effort: ops without a deterministic implementation only warn, so bit-exact results are not
    guaranteed (different GPUs/torch versions/DataLoader worker counts can still differ), and it can be slower.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive numpy/python seeds from the torch worker seed."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_generator(seed: int = 42):
    """Return a seeded ``torch.Generator`` for ``DataLoader(generator=...)``."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
