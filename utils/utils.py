import gc
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple, Any

import numpy as np
from sklearn.base import BaseEstimator


class SharedMemory:
    _shms = []

    def __init__(self, memory: shared_memory.SharedMemory, shape: Tuple[Any, ...], dtype: np.dtype):

        self.shm = memory
        self.shape = shape
        self.dtype = dtype

        SharedMemory.add(self)

    def unlink(self):
        self.shm.close()
        self.shm.unlink()

        if self in SharedMemory._shms:
            SharedMemory._shms.remove(self)

    @staticmethod
    def add(shm):
        SharedMemory._shms.append(shm)

    @staticmethod
    def cleanup():
        for shm in SharedMemory._shms[:]:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"cleanup error: {e}")

        SharedMemory._shms.clear()

    @property
    def array(self):
        return np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self.shm.buf
        )

    def __repr__(self):
        return f"SharedMemory - name={self.shm.name}, shape={self.shape}, dtype={self.dtype}"


def create_shared_numpy(arr: np.ndarray, name: str) -> SharedMemory:
    memory = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    shared_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=memory.buf)
    shared_array[:] = arr[:]
    shared = SharedMemory(memory=memory, shape=arr.shape, dtype=arr.dtype)
    return shared


@dataclass(slots=True)
class TrainData:
    model: Any
    features: SharedMemory = None
    target: SharedMemory = None
    print: bool = False
    train_index: list[int] = None
    test_index: list[int] = None
    featuresIndex: list[int] = None

    @property
    def name(self):
        return self.model.__class__.__name__

    def copy(self):
        return deepcopy(self)

    def set_features_and_targets(self, features: SharedMemory, target: SharedMemory):
        self.features = features
        self.target = target
        if len(self.features.shape) == 1:
            self.featuresIndex = [0]
            return
        self.featuresIndex = list(range(self.features.shape[1]))

    def set_indexes(self, train_index: list[int], test_index: list[int]):
        self.train_index = train_index
        self.test_index = test_index

    def set_features_selection(self, indexes: list[int]):
        self.featuresIndex = indexes

    def __call__(self, train_features: np.ndarray, train_target: np.ndarray):
        self.model.fit(train_features, train_target)
        return self.model


@dataclass(slots=True)
class FeaturesSelectionsData:
    algorithm: BaseEstimator
    features: SharedMemory
    target: SharedMemory
    number_of_features: int


class FilteringCriteria(StrEnum):
    accuracy = "accuracy"
    f1 = "f1"
    recall = "recall"
    precision = "precision"

def create_path(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)  # creates all missing folders
    return p

def force_gc(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect(2)
    return wrapper

def cleanup_shared_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            SharedMemory.cleanup()

    return wrapper