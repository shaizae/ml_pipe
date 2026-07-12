import numpy as np
import pytest
from multiprocessing import shared_memory

from utils.utils import create_shared_numpy, SharedMemory


# adjust imports to your project
# from utils.SharedMemory import SharedMemory, create_shared_numpy


def test_shared_memory_roundtrip(shared_array):
    shm, original = shared_array

    assert np.array_equal(shm.array, original)


def test_shared_memory_write(shared_array):
    shm, _ = shared_array

    shm.array[0, 0] = 999

    assert shm.array[0, 0] == 999


def test_shared_memory_shape(shared_array):
    shm, original = shared_array

    assert shm.shape == original.shape


def test_shared_memory_dtype(shared_array):
    shm, original = shared_array

    assert shm.dtype == original.dtype


def test_shared_memory_repr(shared_array):
    shm, _ = shared_array

    text = repr(shm)

    assert "SharedMemory" in text
    assert shm.shm.name in text


def test_create_shared_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

    shm = create_shared_numpy(arr, "test_create_shared_numpy")

    try:
        assert shm.shape == arr.shape
        assert shm.dtype == arr.dtype
        assert np.array_equal(shm.array, arr)

    finally:
        shm.unlink()


def test_shared_memory_write_is_persistent():
    arr = np.zeros((3, 3), dtype=np.int32)

    shm = create_shared_numpy(arr, "test_persistent_write")

    try:
        shm.array[1, 1] = 123

        assert shm.array[1, 1] == 123

    finally:
        shm.unlink()


def test_shared_memory_unlink_removes_from_registry():
    arr = np.ones((2, 2))

    shm = create_shared_numpy(arr, "test_unlink")

    assert shm in SharedMemory._shms

    shm.unlink()

    assert shm not in SharedMemory._shms


def test_shared_memory_cleanup():
    arr1 = np.ones((2, 2))
    arr2 = np.zeros((3, 3))

    shm1 = create_shared_numpy(arr1, "test_cleanup_1")
    shm2 = create_shared_numpy(arr2, "test_cleanup_2")

    assert len(SharedMemory._shms) > 0

    SharedMemory.cleanup()

    assert len(SharedMemory._shms) == 0


def test_shared_memory_multiple_arrays():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])

    shm1 = create_shared_numpy(arr1, "test_multi_1")
    shm2 = create_shared_numpy(arr2, "test_multi_2")

    try:
        assert np.array_equal(shm1.array, arr1)
        assert np.array_equal(shm2.array, arr2)

    finally:
        shm1.unlink()
        shm2.unlink()