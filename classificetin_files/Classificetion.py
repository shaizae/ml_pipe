import os
from itertools import combinations
from multiprocessing.pool import Pool
from typing import Generator

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from utils.Fetchers import Fetchers
from utils.Results import Results
from utils.Target import Target
from utils.multiprocess_functions import train, features_selections
from utils.utils import create_shared_numpy, TrainData, FeaturesSelectionsData, SharedMemory, FilteringCriteria, \
    cleanup_shared_memory


class Classification:
    _process_limit: int = int(0.8 * os.cpu_count())
    random_state = None

    def __init__(self):
        self._results: list[Results] = None
        self.fetchers: Fetchers = None
        self.targets: Target = None
        self._train_data: list[TrainData] = None

    @staticmethod
    def process_limit(new_limit: int):
        if new_limit < 0:
            raise ValueError('the limit must be greater than 0')
        Classification._process_limit = min(new_limit, os.cpu_count())

    def set(self, fetchers: DataFrame, target: np.ndarray | Series, models: list[BaseEstimator]):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)

        self.targets = Target()
        self.targets.load_new_data(target)

        self._train_data = [TrainData(model) for model in models]

    @cleanup_shared_memory
    def train_test_split(self, ratio: float = 0.8):
        if ratio < 0:
            raise ValueError('ratio must be greater than 0')
        if ratio > 1:
            raise ValueError('ratio must be less than 1')
        target = create_shared_numpy(self.targets.data, "target")
        features = create_shared_numpy(self.fetchers.data, "features")
        train_data = self._train_test_split_generator(ratio, features, target)

        try:
            with Pool(processes=self._process_limit) as pool:
                self._results = list(pool.imap_unordered(train, train_data))

        except Exception as e:
            print(f"training fail error={e}")
            raise e

        return self._results

    def _train_test_split_generator(self, ratio: float, features: SharedMemory, target: SharedMemory) -> Generator[
        TrainData]:
        train_index, test_index = train_test_split(
            np.arange(len(self.fetchers.data)), test_size=ratio)

        for train_data in self._train_data:
            train_data.set_features_and_targets(features, target)
            train_data.print = True
            train_data.set_indexes(train_index, test_index)
            yield train_data

    @cleanup_shared_memory
    def k_folds(self, n_splits: int = 5, shuffle: bool = True):
        if n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")

        if n_splits > len(self.targets.data):
            raise ValueError("n_splits must be less than dataset size")

        results = []
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=Classification.random_state)
        X_train = create_shared_numpy(self.fetchers.data, f"features")
        y_train = create_shared_numpy(self.targets.data, f"target")

        for train_data in self._train_data:
            train_inputs = self._k_fold_generator(train_data, kf, X_train, y_train)
            try:
                with Pool(processes=self._process_limit) as pool:
                    model_results = list(tqdm(pool.imap_unordered(train, train_inputs)))
                final_result = Results(train_data.model)
                for result in tqdm(model_results, desc="uniting_folds"):
                    final_result.append_results(result)
                results.append(final_result)
            except Exception as e:
                print(f"training fail error={e}")

        self._results = results
        return results

    def _k_fold_generator(self, train_data: TrainData, kf: KFold, X_train: SharedMemory, y_train: SharedMemory):
        train_data.set_features_and_targets(X_train, y_train)
        for (train_index, test_index) in kf.split(self.fetchers.data):
            train_inputs = train_data.copy()
            train_inputs.set_indexes(train_index, test_index)
            yield train_inputs

    def leave_one_out(self):
        return self.k_folds(n_splits=len(self.targets.data))

    @cleanup_shared_memory
    def fetcher_selection(self, algorithm: BaseEstimator, number_of_features: list[int]):
        _, train_index = train_test_split(
            np.arange(len(self.fetchers.data)), test_size=0.1)

        fetchers = create_shared_numpy(self.fetchers.pop_index(train_index), "fetchers")
        target = create_shared_numpy(self.targets.pop_index(train_index), "target")

        features_selections_data = self.fetcher_selection_generator(algorithm, number_of_features, fetchers, target)
        try:
            with Pool(processes=self._process_limit) as pool:
                results = list(tqdm(pool.imap_unordered(train, features_selections_data)))
        except Exception as e:
            print(f"fetcher selection fail error={e}")
            raise e
        train_data_list = []
        for train_data_class in tqdm(self._train_data, desc="adding features"):
            for number in results:
                dummy_train_data = train_data_class.copy()
                dummy_train_data.set_features_selection(indexes=number)
                train_data_list.append(dummy_train_data)
        self._train_data = train_data_list

    def fetcher_selection_generator(self, algorithm: BaseEstimator, number_of_features: list[int],
                                    features: SharedMemory, target: SharedMemory):
        for number in number_of_features:
            yield FeaturesSelectionsData(algorithm=algorithm, features=features, target=target,
                                         number_of_features=number)

    @property
    def results(self):
        return self._results

    def filter_by(self, criteria: FilteringCriteria):
        max_value = max(getattr(result, criteria.value) for result in self._results)
        return [item for item in self._results if getattr(item, criteria.value) == max_value]

    def brut_force_features(self):
        number_of_features = range(self.fetchers.data.shape[1])
        vec = np.array(number_of_features)
        all_combinations = list(_brut_force(vec))
        train_data_list = []

        for train_data_class in tqdm(self._train_data, desc="adding features"):
            for combination in all_combinations:
                dummy_train_data = train_data_class.copy()
                dummy_train_data.set_features_selection(indexes=combination)
                train_data_list.append(dummy_train_data)

        self._train_data = train_data_list


def _brut_force(vec: np.ndarray):
    for i in range(len(vec)):
        for combo in combinations(vec, i):
            yield combo
