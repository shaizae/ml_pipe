from pandas import DataFrame, Series
from sklearn.base import BaseEstimator

from Fetchers import Fetchers
from Target import Target


class Classification:
    def __init__(self):
        self.fetchers: Fetchers = None
        self.targets: Target = None
        self.models: list[BaseEstimator] = None

    def load(self, fetchers: DataFrame, target: Series):
        self.fetchers = Fetchers()
        self.targets = Target()
        self.fetchers.load_new_data(fetchers)
        self.targets.load_new_data(target)
