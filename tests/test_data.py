import numpy as np
import scipy.io as sio

from conftest import N_CHANNELS
from emg_gesture import (
    EMGDataset,
    channel_stats,
    extract_group_from_filename,
    extract_label_from_filename,
    list_mat_files,
    load_sample,
    report_eval,
    set_seed,
)


def test_filename_parsing():
    p = "/x/session1_participant13_gesture10_trial2.mat"
    assert extract_label_from_filename(p) == 10
    assert extract_group_from_filename(p) == 13
    assert extract_label_from_filename("/x/foo.mat") is None
    assert extract_group_from_filename("participant3_gesture2_s1.mat") == 3


def test_list_and_load_pad_truncate(mat_dir):
    files = list_mat_files(str(mat_dir))
    assert len(files) == 36
    assert load_sample(files[0], 30).shape == (N_CHANNELS, 30)  # truncate
    x = load_sample(files[0], 80)  # pad
    assert x.shape == (N_CHANNELS, 80) and np.all(x[:, 60:] == 0)
    assert load_sample(files[0], 20, image_shape=(3, 4, 4)).shape == (3, 4, 4)


def test_nan_is_zeroed(tmp_path):
    p = tmp_path / "participant1_gesture1_s1.mat"
    sio.savemat(str(p), {"data_emg": np.array([[np.nan, np.inf], [1.0, -np.inf]])})
    assert np.array_equal(load_sample(str(p), 2), [[0, 1], [0, 0]])


def test_stats_and_dataset_normalisation(mat_dir):
    files = list_mat_files(str(mat_dir))
    mean, std = channel_stats(files, 40)
    assert mean.shape == std.shape == (N_CHANNELS,)
    ds = EMGDataset(files, [0] * len(files), 40, norm_stats=(mean, std))
    xs = np.stack([ds[i][0] for i in range(len(ds))])
    assert len(ds) == len(files) and xs.dtype == np.float32
    np.testing.assert_allclose(xs.mean(axis=(0, 2)), 0, atol=1e-4)
    np.testing.assert_allclose(xs.std(axis=(0, 2)), 1, atol=1e-4)


def test_zero_std_guard(tmp_path):
    p = tmp_path / "participant1_gesture1_s1.mat"
    sio.savemat(str(p), {"data_emg": np.ones((5, 2))})
    mean, std = channel_stats([str(p)], 5)
    assert np.all(std == 1.0) and np.all(mean == 1.0)


def test_report_eval():
    r = report_eval([0, 0, 1, 2], [0, 1, 1, 2], labels=[0, 1, 2], plot=False)
    assert r["accuracy"] == 0.75
    assert r["chance_uniform"] == 1 / 3 and r["chance_majority"] == 0.5
    assert list(r["support"]) == [2, 1, 1]
    assert list(r["per_class_accuracy"]) == [0.5, 1.0, 1.0]
    assert r["confusion_matrix"].shape == (3, 3)


def test_report_eval_plot():
    import matplotlib

    matplotlib.use("Agg")
    assert "fig" in report_eval([0, 1], [0, 1], labels=[0, 1])


def test_set_seed_reproducible():
    set_seed(1)
    a = np.random.rand(3)
    set_seed(1)
    assert np.array_equal(a, np.random.rand(3))


def test_extract_int_is_strict():
    from emg_gesture.data import extract_group_from_filename as grp
    from emg_gesture.data import extract_session_from_filename as sess

    assert sess("/x/session2_participant7_gesture10_trial1.mat") == 2
    assert extract_label_from_filename("gesture10.mat") == 10
    assert sess("/x/mysession2_participant7.mat") is None  # substring no longer matches
    assert sess("/x/session_participant7.mat") is None  # no digits
    assert grp("/x/session1_participant7x_gesture1.mat") is None
    assert extract_label_from_filename("/x/session1_gesturefoo_gesture3.mat") == 3
