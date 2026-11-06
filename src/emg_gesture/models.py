"""SVM baseline pipeline (same steps as the fixed ``svm_*.ipynb`` notebooks)."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ChannelStandardizer(BaseEstimator, TransformerMixin):
    """Per-channel mean/std (over samples and time), fitted on the data passed to ``fit`` only.

    X is (n_samples, target_length * n_channels), flattened from (time, channels).
    Zero-variance channels are guarded (std -> 1).
    """

    def __init__(self, n_channels, eps=1e-8):
        self.n_channels = n_channels
        self.eps = eps

    def fit(self, X, y=None):
        X3 = np.asarray(X).reshape(len(X), -1, self.n_channels)
        mean = np.zeros(self.n_channels)
        std = np.ones(self.n_channels)
        for c in range(self.n_channels):  # one channel at a time to limit memory
            col = X3[:, :, c]
            mean[c] = col.mean(dtype=np.float64)
            std[c] = col.std(dtype=np.float64)
        self.mean_ = mean
        self.std_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, X):
        X3 = np.asarray(X).reshape(len(X), -1, self.n_channels)
        out = (X3 - self.mean_.astype(np.float32)) / self.std_.astype(np.float32)
        return out.reshape(len(X), -1).astype(np.float32, copy=False)


def svm_pipeline(cfg: dict, n_channels: int, seed: int = None) -> Pipeline:
    """ChannelStandardizer -> StandardScaler -> PCA -> SVC from a resolved config.

    Reads ``cfg['svm']`` (pca_components, kernel, C, gamma) and ``cfg['seed']``.
    """
    s = cfg["svm"]
    seed = cfg["seed"] if seed is None else seed
    return Pipeline([
        ("channel_norm", ChannelStandardizer(n_channels=n_channels)),
        ("scaler", StandardScaler(copy=False)),
        ("pca", PCA(n_components=s["pca_components"], random_state=seed)),
        ("svc", SVC(kernel=s["kernel"], C=s["C"], gamma=s["gamma"])),
    ])
