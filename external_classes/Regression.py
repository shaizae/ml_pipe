import os
from itertools import product
from multiprocessing.pool import Pool
from typing import Any, Generator, Iterable

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from results.RegressionResults import RegressionResults
from utils.Fetchers import Fetchers
from utils.Target import Target
from utils.multiprocess_functions import train_k_folds_mc, train_test_split_mc
from utils.utils import TrainData, KFoldsTrainData, SharedMemory, FilteringCriteriaClassification, \
    cleanup_shared_memory, \
    create_shared_numpy, FilteringCriteriaRegression


class Regression:
    _process_limit = max(1, os.cpu_count() // 2)
    random_state = None

    def __init__(self):
        self._features_index: list[list[int]] = []
        self._hyper_parameter: dict[str, list[Any]] = {}
        self._results: list[RegressionResults] = None
        self.fetchers: Fetchers = None
        self.targets: Target = None
        self._train_data: list[TrainData] = None

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 1:
            raise ValueError("limit must be greater than 0")
        Regression._process_limit = min(new_limit, os.cpu_count())

    def set(self, fetchers: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator], ):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)

        self.targets = Target()
        self.targets.load_new_data(target)

        self._features_index = [list(range(self.fetchers.data.shape[1]))]
        self._train_data = [TrainData(model,RegressionResults) for model in models]

    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")

        X = create_shared_numpy(self.fetchers.data, "features")
        y = create_shared_numpy(self.targets.data, "target")

        generator = self._train_test_split_generator(ratio, X, y)

        with Pool(processes=self._process_limit, maxtasksperchild=20, ) as pool:
            self._results = list(
                tqdm(pool.imap_unordered(train_test_split_mc, generator, ), desc="training", ))
        return self._results

    def _train_test_split_generator(self, ratio: float, features: SharedMemory, target: SharedMemory) -> Generator[
        TrainData]:

        train_idx, test_idx = train_test_split(np.arange(len(self.fetchers.data)), test_size=ratio,
                                               random_state=Regression.random_state, )
        for td in self._build_training_list(features, target):
            td.set_indexes(train_idx, test_idx)
            yield td

    @cleanup_shared_memory
    def k_folds(            self,            n_splits: int = 5,            shuffle: bool = True,    ):
        X = create_shared_numpy(self.fetchers.data, "features")
        y = create_shared_numpy(self.targets.data, "target")

        generator = self._k_fold_generator(n_splits, X, y, shuffle, )

        with Pool(processes=self._process_limit, maxtasksperchild=20, ) as pool:
            self._results = list(tqdm(pool.imap_unordered(train_k_folds_mc, generator, ), desc="training", ))

        return self._results

    def leave_one_out(self):
        return self.k_folds(len(self.targets.data))

    def _k_fold_generator(self, k: int, X: SharedMemory, y: SharedMemory, shuffle: bool, ):
        for td in self._build_training_list(X, y):
            yield KFoldsTrainData.set_from_train_data(td, number_of_folds=k, shuffle=shuffle,
                                                      randon_state=Regression.random_state, )

    def _build_training_list(self, X: SharedMemory, y: SharedMemory, ) -> Iterable[TrainData]:

        for base in self._train_data:
            for features in self._features_index:
                td = base.copy()
                td.set_features_and_targets(X, y)
                td.set_features_selection(features)

                for result in self._add_hyper_parameter(td):
                    yield result

    def _add_hyper_parameter(self, train_data: TrainData, ) -> list[TrainData]:
        train_data = train_data.copy()
        params = train_data.get_model_params()

        valid = {k: v for k, v in self._hyper_parameter.items() if k in params}

        if not valid:
            return [train_data]

        keys = list(valid)

        output = []

        for values in product(*(valid[k] for k in keys)):
            td = train_data.copy()
            td.model.set_params(**dict(zip(keys, values)))
            output.append(td)
        return output

    def filter_by(self, criteria: FilteringCriteriaRegression):
        max_value = max(getattr(result, criteria.value) for result in self._results)
        return [item for item in self._results if getattr(item, criteria.value) == max_value]