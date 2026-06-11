from typing import Any

import numpy as np
import pandas as pd
from pandas import Index
from sklearn.preprocessing import StandardScaler, MinMaxScaler,PolynomialFeatures

from tqdm import trange


class Fetchers:
    def __init__(self):
        self._data: np.ndarray = None
        self._columns: Index[str] = None

    def load_new_data(self, data: pd.DataFrame):
        self._data = data.values
        self._columns = data.columns

    def pop_index(self, index: list[int]):
        val = self._data[index, ...]
        self._data = np.delete(self._data, index)
        return val

    @property
    def data(self):
        return self._data

    @property
    def columns(self):
        return self._columns

    def standard_scaler(self, with_mean: Any = True, with_std: Any = True):
        normalizer = StandardScaler(with_mean=with_mean, with_std=with_std)
        self._data = normalizer.fit_transform(self._data)
        return self.data

    def min_max_scaler(self, min_val=0, max_val=1):
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        scaler = MinMaxScaler(feature_range=(min_val, max_val))
        self._data = scaler.fit_transform(self._data)
        return self.data

    def __iter__(self):
        for row in trange(self._data.shape[0]):
            yield self._data[row, ...]

    def min(self):
        return np.min(self._data)

    def max(self):
        return np.max(self._data)

    def mean(self):
        return np.mean(self._data)

    def std(self):
        return np.std(self._data)

    def median(self):
        return np.median(self._data)

    def polynomial_features(self,degree: int=2,include_bias:bool=False) :
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self._data=polynomial_features.fit_transform(self._data)
        return self.data


