"""Handcrafted time-domain EMG features and the features -> StandardScaler -> classifier pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ALL_FEATURES = ("rms", "mav", "wl", "zc", "ssc", "var")
CLASSIFIERS = ("lda", "rf", "svm")


def _rms(x):
    return np.sqrt(np.mean(x * x, axis=1))


def _mav(x):
    return np.mean(np.abs(x), axis=1)


def _wl(x):
    return np.sum(np.abs(np.diff(x, axis=1)), axis=1)


def _zc(x):
    return np.sum(x[:, :-1] * x[:, 1:] < 0, axis=1).astype(np.float64)


def _ssc(x):
    d1 = x[:, 1:-1] - x[:, :-2]
    d2 = x[:, 1:-1] - x[:, 2:]
    return np.sum(d1 * d2 > 0, axis=1).astype(np.float64)


def _var(x):
    return np.var(x, axis=1)


_FUNCS = {"rms": _rms, "mav": _mav, "wl": _wl, "zc": _zc, "ssc": _ssc, "var": _var}


class EMGFeatureExtractor(BaseEstimator, TransformerMixin):
    """Per-channel time-domain features. Stateless: nothing is fitted on data.

    X is (n_samples, target_length * n_channels), flattened from (time, channels). Output is
    (n_samples, n_channels * n_features), channel-major: channel 0's features first.
    ZC counts sign changes between neighbouring samples; SSC counts slope sign changes (both
    without a threshold); var is the population variance.
    """

    def __init__(self, n_channels, features=ALL_FEATURES):
        self.n_channels = n_channels
        self.features = features

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ch{c}_{f}" for c in range(self.n_channels) for f in self.features], dtype=object)

    def transform(self, X):
        unknown = [f for f in self.features if f not in _FUNCS]
        if unknown:
            raise ValueError(f"unknown features {unknown}; choose from {list(ALL_FEATURES)}")
        X = np.asarray(X, dtype=np.float64)
        X3 = X.reshape(len(X), -1, self.n_channels)
        cols = []
        for c in range(self.n_channels):
            x = X3[:, :, c]
            cols.extend(_FUNCS[f](x) for f in self.features)
        return np.stack(cols, axis=1)


def features_pipeline(cfg: dict, n_channels: int, seed: int = None) -> Pipeline:
    """EMGFeatureExtractor -> StandardScaler -> classifier from ``cfg['features']`` and ``cfg['seed']``."""
    s = cfg["features"]
    unknown = [f for f in s["features"] if f not in _FUNCS]
    if unknown:
        raise ValueError(f"unknown features {unknown}; choose from {list(ALL_FEATURES)}")
    seed = cfg["seed"] if seed is None else seed
    name = s.get("classifier", "lda")
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=int(s["rf"]["n_estimators"]), random_state=seed, n_jobs=1)
    elif name == "svm":
        clf = SVC(C=s["svm"]["C"], kernel=s["svm"]["kernel"])
    else:
        raise ValueError(f"unknown classifier {name!r}; choose from {list(CLASSIFIERS)}")
    return Pipeline([
        ("features", EMGFeatureExtractor(n_channels=n_channels, features=tuple(s["features"]))),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
