from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder


class Target:
    def __init__(self):
        self._data: np.ndarray = None
        self.decrypt_map: dict[Any, int] = None

    def load_new_data(self, target: np.ndarray):
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(target)
        mapping = {
            label: int(index)
            for index, label in enumerate(encoder.classes_)
        }
        self._data = encoded
        self.decrypt_map = mapping

    @property
    def data(self):
        return self._data
