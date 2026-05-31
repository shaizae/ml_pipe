import os
from multiprocessing.pool import Pool

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Fetchers import Fetchers
from Results import Results
from Target import Target
from utils import create_shared_numpy, TrainData


def train(train_data: TrainData):
    print(f"train start whit model: {train_data.name}")
    train_data.model.fit(train_data.features.array, train_data.target.array)
    print(f"train end whit model: {train_data.name}")
    return Results(train_data.model)


class Classification:
    _process_limit = os.cpu_count()

    def __init__(self):
        self.fetchers: Fetchers = None
        self.targets: Target = None
        self.models: list[BaseEstimator] = None

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

        self.models = models

    def train_test_split(self, ratio: float = 0.8):
        if ratio < 0:
            raise ValueError('ratio must be greater than 0')
        if ratio > 1:
            raise ValueError('ratio must be less than 1')

        X_train, X_test, y_train, y_test = train_test_split(
            self.fetchers.data.values, self.targets.data, test_size=ratio
        )
        targe = create_shared_numpy(y_train, "targe")
        features = create_shared_numpy(X_train, "features")

        data: list[TrainData] = []
        for model in self.models:
            data.append(TrainData(model, features, targe))

        pool = Pool(processes=self._process_limit)
        results = pool.map(train, data)
        pool.close()
        pool.join()

        features.unlink()
        targe.unlink()

        for result in tqdm(results, desc="add test data to results"):
            result.set_test(X_test, y_test)

        return results
