import numpy as np


def test_shared_memory_roundtrip(shared_array):
    shm, original = shared_array

    loaded = shm.array

    assert np.array_equal(loaded, original)


def test_shared_memory_write(shared_array):
    shm, _ = shared_array

    shm.array[0, 0] = 999

    assert shm.array[0, 0] == 999


def test_shared_memory_shape(shared_array):
    shm, original = shared_array

    assert shm.shape == original.shape
