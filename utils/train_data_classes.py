from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from results import BaseResults
from utils.utils import SharedMemory


@dataclass(slots=True)
class TrainData:
    model: Any
    results_type: BaseResults
    features: SharedMemory = None
    target: SharedMemory = None
    train_index: list[int] = None
    test_index: list[int] = None
    featuresIndex: list[int] = None


    @property
    def name(self):
        return self.model.__class__.__name__

    def get_model_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.model.__dict__.items() if not k.startswith("_")}

    def copy(self):
        return deepcopy(self)

    def set_features_and_targets(self, features: SharedMemory, target: SharedMemory):
        self.features = features
        self.target = target

    def set_indexes(self, train_index: list[int], test_index: list[int]):
        self.train_index = train_index
        self.test_index = test_index

    def set_features_selection(self, indexes: list[int]):
        self.featuresIndex = indexes

    def fit(self, train_features: np.ndarray, train_target: np.ndarray):
        self.model.fit(train_features, train_target)
        return self.model

    def __iter__(self):
        for field in fields(self):
            yield field.name, getattr(self, field.name)


@dataclass(slots=True)
class KFoldsTrainData(TrainData):
    number_of_folds: int = 0
    shuffle: bool = True
    randon_state: int = None

    @staticmethod
    def set_from_train_data(train_data: TrainData, number_of_folds: int, shuffle: bool,
                            randon_state: int, ) -> KFoldsTrainData:
        return KFoldsTrainData(**dict(train_data), number_of_folds=number_of_folds, shuffle=shuffle,
                               randon_state=randon_state, )
