from abc import ABC, abstractmethod
from typing import Any


class BaseML(ABC):

    def __init__(self):
        self._features_index: list[list[int]] = None
        self._hyper_parameter: dict[str, list[Any]] = {}
        self.fetchers = None
        self.targets = None
        self._train_data = None

    @abstractmethod
    def set(self, fetchers, target, models):
        ...

    @abstractmethod
    def train_test_split(self, ratio=0.8):
        ...

    @abstractmethod
    def k_folds(self, n_splits=5, shuffle=True):
        ...

    @abstractmethod
    def leave_one_out(self):
        ...

    @abstractmethod
    def filter_by(self, criteria):
        ...