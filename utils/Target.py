from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder


class Target:
    def __init__(self):
        self._data: np.ndarray = None
        self.decrypt_map: dict[Any, int] = None
        self._encoder: LabelEncoder = None

    def load_new_data(self, target: np.ndarray):
        self._encoder = LabelEncoder()
        encoded = self._encoder.fit_transform(target)
        mapping = {
            label: int(index)
            for index, label in enumerate(self._encoder.classes_)
        }
        self._data = encoded
        self.decrypt_map = mapping

    @property
    def data(self):
        return self._data

    @property
    def show_origial_data(self):
        if self._data is None:
            return None

        return self._encoder.inverse_transform(self._data)
