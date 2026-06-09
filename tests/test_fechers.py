import pandas as pd

from utils.Fetchers import Fetchers


def test_load_new_data():
    fetchers = Fetchers()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    fetchers.load_new_data(data)

    assert fetchers.data is data


def test_data_property_returns_dataframe():
    fetchers = Fetchers()

    data = pd.DataFrame({
        "feature1": [10, 20],
        "feature2": [30, 40]
    })

    fetchers.load_new_data(data)

    assert isinstance(fetchers.data, pd.DataFrame)


def test_data_shape_is_preserved():
    fetchers = Fetchers()

    data = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    fetchers.load_new_data(data)

    assert fetchers.data.shape == (3, 2)


def test_overwrite_existing_data():
    fetchers = Fetchers()

    first = pd.DataFrame({"a": [1, 2]})
    second = pd.DataFrame({"b": [3, 4]})

    fetchers.load_new_data(first)
    fetchers.load_new_data(second)

    assert fetchers.data is second
    assert list(fetchers.data.columns) == ["b"]


def test_pop_feature():
    fetchers = Fetchers()
    data = pd.DataFrame({"feature1": list(range(1, 50)), "feature2": list(range(20, 69)), })
    fetchers.load_new_data(data)
    popped = fetchers.pop_index([1, 6, 9, 7, 15])
    expected_popped = data.loc[[1, 6, 9, 7, 15]]
    pd.testing.assert_frame_equal(popped, expected_popped)
    expected_remaining = data.drop([1, 6, 9, 7, 15])
    pd.testing.assert_frame_equal(fetchers.data.sort_index(), expected_remaining.sort_index())
