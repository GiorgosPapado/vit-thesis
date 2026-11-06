import numpy as np

from emg_gesture.augment import add_noise, augment_data, random_crop_pad, scale_signal, time_warp


def _x():
    return np.arange(40, dtype=float).reshape(10, 4)


def test_reproducible_with_same_seed_and_differs_across_seeds():
    for fn in (add_noise, time_warp, scale_signal, augment_data):
        a = fn(_x(), np.random.default_rng(1))
        b = fn(_x(), np.random.default_rng(1))
        c = fn(_x(), np.random.default_rng(2))
        np.testing.assert_array_equal(a, b)
        assert a.shape != c.shape or not np.array_equal(a, c)


def test_does_not_touch_global_numpy_state():
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    augment_data(_x(), np.random.default_rng(0))
    np.testing.assert_array_equal(before, np.random.get_state()[1])


def test_scale_and_noise_bounds():
    x = np.ones((5, 3))
    for s in range(20):
        y = scale_signal(x, np.random.default_rng(s), 0.1)
        assert 0.9 <= y[0, 0] <= 1.1 and np.allclose(y, y[0, 0])
    assert np.allclose(add_noise(x, np.random.default_rng(0), 0.0), x)


def test_time_warp_length_and_values():
    y = time_warp(_x(), np.random.default_rng(0), 0.1)
    assert 9 <= len(y) <= 11 and set(y.ravel()) <= set(_x().ravel())


def test_random_crop_pad():
    rng = np.random.default_rng(0)
    padded = random_crop_pad(np.ones((6, 2)), rng, 10)
    assert padded.shape == (10, 2) and padded.sum() == 12
    assert random_crop_pad(_x(), rng, 4).shape == (4, 4)
    np.testing.assert_array_equal(random_crop_pad(_x(), rng, 10), _x())
