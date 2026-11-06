"""AugmentedDataset RNG behaviour, tested with a torch-free fake base dataset and worker context."""
import numpy as np

from emg_gesture.train_vit import AugmentedDataset


class Base:
    def __len__(self):
        return 3

    def __getitem__(self, i):
        return np.ones((8, 4), dtype=np.float32), i


class Fake(AugmentedDataset):
    ctx = (None, None)

    def _worker_context(self):
        return self.ctx


def _draw(ds):
    return ds[0][0]


def test_epochs_do_not_replay_noise():
    ds = Fake(Base(), seed=1)
    a = _draw(ds)
    ds.set_epoch(1)
    b = _draw(ds)
    assert not np.array_equal(a, b)


def test_reproducible_for_same_seed_and_epoch():
    a, b = Fake(Base(), seed=1), Fake(Base(), seed=1)
    a.set_epoch(2), b.set_epoch(2)
    np.testing.assert_array_equal(_draw(a), _draw(b))


def test_workers_and_worker_seeds_differ():
    outs = []
    for ctx in ((None, None), (0, 100), (1, 100), (0, 101)):
        ds = Fake(Base(), seed=1)
        ds.ctx = ctx
        outs.append(_draw(ds))
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert not np.array_equal(outs[i], outs[j])


def test_rng_advances_within_epoch():
    ds = Fake(Base(), seed=1)
    assert not np.array_equal(_draw(ds), _draw(ds))
