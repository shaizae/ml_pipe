import os
from abc import ABC, abstractmethod
from itertools import combinations, product
from typing import Any, Iterable

import numpy as np

from utils.train_data_classes import TrainData


class BaseML(ABC):
    _process_limit: int = max(1, os.cpu_count() // 2)
    random_state = None

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

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 1:
            raise ValueError("limit must be greater than 0")
        BaseML._process_limit = min(new_limit, os.cpu_count())

    def set_hyper_parameters(self, hyper_parameter: dict[str, list[Any]]):
        self._hyper_parameter = hyper_parameter

    @abstractmethod
    def filter_by(self, criteria):
        values = [getattr(result, criteria.value) for result in self._results]

        best_value = max(values) if criteria.maximize else min(values)

        return [result for result in self._results if getattr(result, criteria.value) == best_value]


def _brut_force(vec: np.ndarray) -> Iterable[np.ndarray]:
    for i in range(1, len(vec)):
        for combo in combinations(vec, i):
            yield combo


def _add_hyper_parameter(_hyper_parameter, data: TrainData) -> list[TrainData]:
    train_data = data.copy()
    model_params = train_data.get_model_params()
    valid_params = {k: v for k, v in _hyper_parameter.items() if k in model_params}
    if not valid_params:
        return [train_data]
    keys = list(valid_params.keys())
    new_train_data = []
    for values in product(*(valid_params[k] for k in keys)):
        td = train_data.copy()
        params = dict(zip(keys, values))
        td.model.set_params(**params)
        new_train_data.append(td)
    return new_train_data
