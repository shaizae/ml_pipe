import numpy as np
import pytest

# Change this import to your actual file location
from utils.utils import     SharedMemory,    create_shared_numpy,    cleanup_shared_memory



@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Prevent shared memory leaks between tests.
    """
    yield
    SharedMemory.cleanup()


def test_shared_memory_roundtrip():
    original = np.array(
        [[1, 2], [3, 4]],
        dtype=np.int32
    )

    shm = create_shared_numpy(
        original,
        "test_roundtrip"
    )

    try:
        loaded = shm.array

        assert np.array_equal(loaded, original)

    finally:
        shm.unlink()


def test_shared_memory_write():
    original = np.zeros(
        (3, 3),
        dtype=np.int32
    )

    shm = create_shared_numpy(
        original,
        "test_write"
    )

    try:
        shm.array[0, 0] = 999

        assert shm.array[0, 0] == 999

    finally:
        shm.unlink()


def test_shared_memory_shape():
    original = np.zeros(
        (5, 7),
        dtype=np.float32
    )

    shm = create_shared_numpy(
        original,
        "test_shape"
    )

    try:
        assert shm.shape == original.shape

    finally:
        shm.unlink()


def test_shared_memory_dtype():
    original = np.zeros(
        (5, 5),
        dtype=np.float64
    )

    shm = create_shared_numpy(
        original,
        "test_dtype"
    )

    try:
        assert shm.dtype == original.dtype

    finally:
        shm.unlink()


def test_shared_memory_repr():
    original = np.zeros((2, 2))

    shm = create_shared_numpy(
        original,
        "test_repr"
    )

    try:
        text = repr(shm)

        assert "SharedMemory" in text
        assert shm.shm.name in text

    finally:
        shm.unlink()


# -------------------------
# SharedMemory lifecycle
# -------------------------

def test_shared_memory_is_registered():
    original = np.ones((2, 2))

    shm = create_shared_numpy(
        original,
        "test_registered"
    )

    try:
        assert shm in SharedMemory._shms

    finally:
        shm.unlink()

def test_shared_memory_unlink_removes_registry():
    original = np.ones((2, 2))

    shm = create_shared_numpy(
        original,
        "test_unlink"
    )

    assert shm in SharedMemory._shms

    shm.unlink()

    assert shm not in SharedMemory._shms


def test_shared_memory_cleanup():
    create_shared_numpy(
        np.ones((2, 2)),
        "test_cleanup_1"
    )

    create_shared_numpy(
        np.zeros((3, 3)),
        "test_cleanup_2"
    )

    assert len(SharedMemory._shms) > 0

    SharedMemory.cleanup()

    assert SharedMemory._shms == []


def test_multiple_shared_memory_objects():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])

    shm1 = create_shared_numpy(
        arr1,
        "test_multiple_1"
    )

    shm2 = create_shared_numpy(
        arr2,
        "test_multiple_2"
    )

    try:
        assert np.array_equal(shm1.array, arr1)
        assert np.array_equal(shm2.array, arr2)

    finally:
        shm1.unlink()
        shm2.unlink()


# -------------------------
# cleanup_shared_memory decorator tests
# -------------------------

def test_cleanup_decorator_after_success():

    @cleanup_shared_memory
    def create_memory():
        arr = np.ones((10, 10))

        shm = create_shared_numpy(
            arr,
            "decorator_success"
        )

        assert len(SharedMemory._shms) > 0

        return shm.array.copy()

    result = create_memory()

    assert np.array_equal(
        result,
        np.ones((10, 10))
    )

    assert SharedMemory._shms == []


def test_cleanup_decorator_after_exception():

    @cleanup_shared_memory
    def failing_function():

        create_shared_numpy(
            np.zeros((5, 5)),
            "decorator_exception"
        )

        assert len(SharedMemory._shms) > 0

        raise RuntimeError("expected error")

    with pytest.raises(
        RuntimeError,
        match="expected error"
    ):
        failing_function()

    assert SharedMemory._shms == []


def test_cleanup_decorator_keeps_return_value():

    @cleanup_shared_memory
    def return_value():

        create_shared_numpy(
            np.array([1, 2, 3]),
            "decorator_return"
        )

        return 42

    result = return_value()

    assert result == 42
    assert SharedMemory._shms == []


def test_cleanup_decorator_passes_arguments():

    @cleanup_shared_memory
    def create_with_argument(size):

        arr = np.zeros((size, size))

        create_shared_numpy(
            arr,
            "decorator_arguments"
        )

        return arr.shape

    result = create_with_argument(7)

    assert result == (7, 7)
    assert SharedMemory._shms == []


def test_cleanup_decorator_multiple_shared_memory():

    @cleanup_shared_memory
    def create_multiple():

        create_shared_numpy(
            np.ones((2, 2)),
            "decorator_multi_1"
        )

        create_shared_numpy(
            np.zeros((3, 3)),
            "decorator_multi_2"
        )

        assert len(SharedMemory._shms) > 0

    create_multiple()

    assert SharedMemory._shms == []