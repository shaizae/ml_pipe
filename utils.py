from dataclasses import dataclass
from multiprocessing import shared_memory,Lock
from typing import Tuple, Any

import numpy as np


class SharedMemory:
    def __init__(self, memory: shared_memory.SharedMemory, shape: Tuple[Any, ...], dtype: np.dtype):
        self.shm = memory
        self.shape = shape
        self.dtype = dtype

    def unlink(self):
        self.shm.close()
        self.shm.unlink()

    @property
    def array(self):
        return np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self.shm.buf
        )


def create_shared_numpy(arr: np.ndarray, name: str) -> SharedMemory:
    """
    Create shared memory from a NumPy array.

    Returns:
        shm: SharedMemory object
        shape: original shape
        dtype: original dtype
    """
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]
    return SharedMemory(shm, arr.shape, arr.dtype)


@dataclass(slots=True)
class TrainData:
    model: Any
    features: SharedMemory
    target: SharedMemory

    @property
    def name(self):
        return self.model.__class__.__name__
