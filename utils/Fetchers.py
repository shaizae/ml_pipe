from pandas import DataFrame


class Fetchers:
    def __init__(self):
        self._data:DataFrame=None

    def load_new_data(self, data):
        self._data = data

    @property
    def data(self):
        return self._data