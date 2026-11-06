import numpy as np
import pytest

from conftest import N_CHANNELS
from emg_gesture import (
    extract_label_from_filename,
    list_mat_files,
    load_flattened,
    remap_labels,
    split_files,
)
from emg_gesture.models import ChannelStandardizer, svm_pipeline

T = 40
CFG = {"seed": 0, "svm": {"pca_components": 5, "kernel": "linear", "C": 1.0, "gamma": "scale"}}


def _xy(mat_dir):
    files = list_mat_files(str(mat_dir))
    X = np.array([load_flattened(f, T) for f in files])
    y = np.array([extract_label_from_filename(f) for f in files])
    return files, X, y


def test_pipeline_fits_and_predicts(mat_dir):
    _, X, y = _xy(mat_dir)
    pipe = svm_pipeline(CFG, N_CHANNELS).fit(X, y)
    assert X.shape == (36, T * N_CHANNELS)
    assert pipe.predict(X).shape == y.shape
    assert [n for n, _ in pipe.steps] == ["channel_norm", "scaler", "pca", "svc"]
    assert pipe.named_steps["pca"].n_components_ == 5
    assert pipe.named_steps["svc"].kernel == "linear"


def test_kernel_and_seed_from_config():
    cfg = {"seed": 7, "svm": {"pca_components": 3, "kernel": "rbf", "C": 2.0, "gamma": "scale"}}
    pipe = svm_pipeline(cfg, 4)
    assert pipe.named_steps["svc"].kernel == "rbf" and pipe.named_steps["svc"].C == 2.0
    assert pipe.named_steps["pca"].random_state == 7


def test_channel_standardizer_per_channel_and_zero_variance():
    rng = np.random.default_rng(0)
    X3 = rng.normal(size=(10, 6, 3)) * np.array([1.0, 5.0, 0.0]) + np.array([0.0, 10.0, 2.0])
    X = X3.reshape(10, -1)
    out = ChannelStandardizer(n_channels=3).fit_transform(X).reshape(10, 6, 3)
    assert np.allclose(out[:, :, :2].mean(axis=(0, 1)), 0, atol=1e-5)
    assert np.allclose(out[:, :, :2].std(axis=(0, 1)), 1, atol=1e-5)
    assert np.all(np.isfinite(out)) and np.allclose(out[:, :, 2], 0)  # constant channel: std guard


def test_stats_come_from_fit_data_only():
    """Statistics are fitted on the train data; transforming val/test never changes them."""
    rng = np.random.default_rng(1)
    train = rng.normal(size=(8, 12))
    test = rng.normal(loc=100, size=(4, 12))
    t = ChannelStandardizer(n_channels=2).fit(train)
    mean, std = t.mean_.copy(), t.std_.copy()
    t.transform(test)
    assert np.array_equal(mean, t.mean_) and np.array_equal(std, t.std_)
    assert np.allclose(mean, train.reshape(8, -1, 2).mean(axis=(0, 1)))
    # a transformer fitted on test would differ from the one fitted on train
    assert not np.allclose(ChannelStandardizer(n_channels=2).fit(test).mean_, mean)


def test_no_leakage_scaler_and_pca_fitted_on_train_only(mat_dir):
    files, X, y = _xy(mat_dir)
    train, _, test = split_files(files, y, mode="grouped", val_size=0.0, seed=0)
    pipe = svm_pipeline(CFG, N_CHANNELS).fit(X[train], y[train])
    ct = ChannelStandardizer(n_channels=N_CHANNELS).fit(X[train])
    assert np.allclose(pipe.named_steps["channel_norm"].mean_, ct.mean_)
    # scaler and PCA mean reflect the training rows only
    Z = ct.transform(X[train])
    assert np.allclose(pipe.named_steps["scaler"].mean_, Z.mean(axis=0), atol=1e-5)
    Zs = pipe.named_steps["scaler"].transform(Z)
    assert np.allclose(pipe.named_steps["pca"].mean_, Zs.mean(axis=0), atol=1e-4)
    assert pipe.named_steps["scaler"].n_samples_seen_ == len(train)
    assert pipe.named_steps["pca"].n_samples_ == len(train)
    # predicting on the test split must not alter any fitted state
    before = pipe.named_steps["scaler"].mean_.copy()
    pipe.predict(X[test])
    assert np.array_equal(before, pipe.named_steps["scaler"].mean_)


def test_pca_refitted_inside_each_cv_fold(mat_dir):
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    _, X, y = _xy(mat_dir)
    pipe = svm_pipeline(CFG, N_CHANNELS)
    for tr, _ in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        fold = clone(pipe).fit(X[tr], y[tr])
        assert fold.named_steps["pca"].n_samples_ == len(tr) < len(X)


def test_remap_labels():
    mapped, classes = remap_labels([10, 3, 7, 3])
    assert list(classes) == [3, 7, 10] and list(mapped) == [2, 0, 1, 0]
    again, _ = remap_labels([7, 10], classes)
    assert list(again) == [1, 2]
    assert [classes[i] for i in mapped] == [10, 3, 7, 3]  # invertible
    with pytest.raises(ValueError):
        remap_labels([4], classes)
    with pytest.raises(ValueError):
        remap_labels([11], classes)
