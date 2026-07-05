from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import trange


class Target:
    def __init__(self):
        self._data: Optional[np.ndarray] = None
        self.decrypt_map: Optional[dict[Any, int]] = None
        self._encoder: Optional[LabelEncoder] = None

    def load_new_data(self, target: np.ndarray):
        self._encoder = LabelEncoder()
        encoded = self._encoder.fit_transform(target)
        mapping = {
            label: int(index)
            for index, label in enumerate(self._encoder.classes_)
        }
        self._data = encoded
        self.decrypt_map = mapping

    def pop_index(self, index: list[int]):
        val = self._data[index]
        self._data = np.delete(self._data, index)
        return val

    @property
    def data(self):
        return self._data

    @property
    def show_original_data(self):
        if self._data is None:
            return None

        return self._encoder.inverse_transform(self._data)

    def __iter__(self):
        for row in trange(self._data.shape[0]):
            yield self._data[row]
