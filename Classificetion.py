import os
from multiprocessing.pool import Pool

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm

from Fetchers import Fetchers
from Results import Results
from Target import Target
from utils import create_shared_numpy, TrainData


def train(train_data: TrainData):
    print(f"train start whit model: {train_data.name}") if train_data.print else None
    features = train_data.features.array
    target = train_data.target.array
    train_features = features[train_data.train_index,...]
    train_target = target[train_data.train_index]
    train_data.model.fit(train_features, train_target)
    print(f"train end whit model: {train_data.name}") if train_data.print else None
    result=Results(train_data.model)
    test_features = features[train_data.test_index, ...]
    test_target = target[train_data.test_index]
    result.set_test(test_features, test_target)
    return result


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

        train_index, test_index= train_test_split(
            np.arange(len(self.fetchers.data)), test_size=ratio)
        targe = create_shared_numpy(self.targets.data, "targe")
        features = create_shared_numpy(self.fetchers.data.to_numpy(), "features")

        for train_data in self._train_data:
            train_data.set_features_and_targets(features, targe)
            train_data.print = True
            train_data.set_indexes(train_index,test_index)

        with Pool(processes=self._process_limit) as pool:
            results = pool.map(train, self._train_data)

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
                train_input=train_data.copy()
                train_input.set_indexes(train_index, test_index)
                train_inputs.append(train_input)

            with Pool(processes=self._process_limit) as pool:
                model_results = pool.map(train, train_inputs)
            results.extend(model_results)
        X_train.unlink()
        y_train.unlink()
        return results
