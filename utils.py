from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import shared_memory
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
    features: SharedMemory = None
    target: SharedMemory = None
    print: bool = False
    train_index: list[int] = None
    test_index: list[int] = None

    @property
    def name(self):
        return self.model.__class__.__name__

    def copy(self):
        return deepcopy(self)

    def set_features_and_targets(self, features: SharedMemory, target: SharedMemory):
        self.features = features
        self.target = target

    def set_indexes(self, train_index: list[int], test_index: list[int]):
        self.train_index = train_index
        self.test_index = test_index

    @property
    def have_indexes(self):
        return self.features_index is not None and self.targets_index is not None

    def __call__(self, train_features:np.ndarray, train_target:np.ndarray):
        self.model.fit(train_features, train_target)
        return self.model
