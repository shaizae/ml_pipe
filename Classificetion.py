import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

from Fetchers import Fetchers
from Target import Target


class Classification:
    def __init__(self):
        self.fetchers: Fetchers = None
        self.targets: Target = None
        self.models: list[BaseEstimator] = None

    def load(self, fetchers: DataFrame, target: np.ndarray| Series):
        if isinstance(target, Series):
            target = target.to_numpy(copy=True)

        self.fetchers = Fetchers()
        self.fetchers.load_new_data(fetchers)

        self.targets = Target()
        self.targets.load_new_data(target)
