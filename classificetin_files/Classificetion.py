import os
from multiprocessing.pool import Pool

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from utils.Fetchers import Fetchers
from utils.Results import Results
from utils.Target import Target
from utils.ultyprosses_functions import train, features_selections
from utils.utils import create_shared_numpy, TrainData, FeaturesSelectionsData


class Classification:
    _process_limit = os.cpu_count()

    def __init__(self):
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

    def train_test_split(self, ratio: float = 0.8):
        if ratio < 0:
            raise ValueError('ratio must be greater than 0')
        if ratio > 1:
            raise ValueError('ratio must be less than 1')

        train_index, test_index = train_test_split(
            np.arange(len(self.fetchers.data)), test_size=ratio)
        targe = create_shared_numpy(self.targets.data, "targe")
        features = create_shared_numpy(self.fetchers.data.to_numpy(), "features")

        for train_data in self._train_data:
            train_data.set_features_and_targets(features, targe)
            train_data.print = True
            train_data.set_indexes(train_index, test_index)

        with Pool(processes=self._process_limit) as pool:
            runner = pool.map_async(train, self._train_data)
            results = runner.get()

        features.unlink()
        targe.unlink()
        return results

    def k_folds(self, n_splits: int = 5):
        if n_splits <= 1:
            raise ValueError("n_splits must be greater than 1")

        if n_splits > len(self.targets.data):
            raise ValueError("n_splits must be less than dataset size")

        results = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        X_train = create_shared_numpy(self.fetchers.data.to_numpy(), f"features")
        y_train = create_shared_numpy(self.targets.data, f"target")
        bar = tqdm(self._train_data)

        for train_data in bar:
            bar.set_description(f"model {train_data.name}")
            train_data.set_features_and_targets(X_train, y_train)

            train_inputs = []

            for fold, (train_index, test_index) in enumerate(kf.split(self.fetchers.data), start=1):
                train_input = train_data.copy()
                train_input.set_indexes(train_index, test_index)
                train_inputs.append(train_input)

            with Pool(processes=self._process_limit) as pool:
                runner = pool.map_async(train, train_inputs)
                model_results = runner.get()

            final_result = Results(train_data.model)
            for result in tqdm(model_results):
                final_result.append_results(result)
            results.append(final_result)

        X_train.unlink()
        y_train.unlink()
        return results

    def leave_one_out(self):
        return self.k_folds(n_splits=len(self.targets.data))

    def fetcher_selection(self, algorithm: BaseEstimator, number_of_features: list[int]):
        _, train_index = train_test_split(
            np.arange(len(self.fetchers.data)), test_size=0.1)

        fetchers = create_shared_numpy(self.fetchers.pop_index(train_index).values, "fetchers")
        target = create_shared_numpy(self.targets.pop_index(train_index), "target")

        features_selections_data = [FeaturesSelectionsData] * len(number_of_features)
        for ind, number in enumerate(tqdm(number_of_features,desc="selecting features")):
            features_selections_data[ind] = FeaturesSelectionsData(algorithm=algorithm, features=fetchers,
                                                                   target=target, number_of_features=number)

        with Pool(processes=self._process_limit) as pool:
            runner = pool.map_async(features_selections, features_selections_data)
            results = runner.get()
        train_data_list=[]
        for train_data_class in tqdm(self._train_data,desc="adding features"):
            for number in results:
                dummy_train_data=train_data_class.copy()
                dummy_train_data.set_features_selection(indexes=number)
                train_data_list.append(dummy_train_data)
        self._train_data = train_data_list





