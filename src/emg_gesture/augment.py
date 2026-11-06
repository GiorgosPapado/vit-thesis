"""Signal augmentations ported from ``vit_train_from_scratch_10gest.ipynb``.

All functions take a ``numpy.random.Generator`` (never global numpy state) so runs are reproducible.
They act on arrays whose first axis is time for ``time_warp``/``random_crop_pad``; ``add_noise`` and
``scale_signal`` work on any shape (the notebook applies them to the normalised (3, 224, 224) image).
"""

from __future__ import annotations

import numpy as np


def add_noise(x, rng: np.random.Generator, noise_factor: float = 0.05):
    """Add Gaussian noise with standard deviation ``noise_factor``."""
    return x + rng.standard_normal(np.shape(x)) * noise_factor


def time_warp(x, rng: np.random.Generator, time_warp_factor: float = 0.1):
    """Stretch/compress axis 0 by a random factor in [1-f, 1+f] (nearest-index resampling).

    The output length is ``int(len * factor)``, so it can differ from the input length.
    """
    n = x.shape[0]
    stretch = rng.uniform(1 - time_warp_factor, 1 + time_warp_factor)
    indices = np.round(np.linspace(0, n - 1, int(n * stretch))).astype(int)
    return x[indices % n]


def scale_signal(x, rng: np.random.Generator, scale_factor: float = 0.1):
    """Multiply by one random factor in [1-s, 1+s]."""
    return x * rng.uniform(1 - scale_factor, 1 + scale_factor)


def random_crop_pad(x, rng: np.random.Generator, target_length: int):
    """Zero-pad (random offset) or randomly crop axis 0 to ``target_length``.

    Unlike the notebook (``randint(0, k)`` excludes ``k`` and fails when ``k == 0``), offsets here
    are inclusive of the extreme positions and an exact-length input is returned unchanged.
    """
    n = x.shape[0]
    if n < target_length:
        pad = target_length - n
        before = int(rng.integers(0, pad + 1))
        widths = ((before, pad - before),) + ((0, 0),) * (x.ndim - 1)
        return np.pad(x, widths, mode="constant")
    start = int(rng.integers(0, n - target_length + 1))
    return x[start:start + target_length]


def augment_data(x, rng: np.random.Generator, noise_factor: float = 0.05, scale_factor: float = 0.1):
    """The notebook's active pipeline: noise then scaling (time warp, crop/pad, flip were commented out)."""
    return scale_signal(add_noise(x, rng, noise_factor), rng, scale_factor)
