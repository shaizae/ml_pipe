from pandas import DataFrame


class Fetchers:
    def __init__(self):
        self._data:DataFrame=None

    def load_new_data(self, data):
        self._data = data
    def pop_index(self,index: list[int]):
        rows = self._data.loc[index].copy()  # save the row
        self._data = self._data.drop(index)
        return rows

    @property
    def data(self):
        return self._data