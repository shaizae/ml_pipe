import numpy as np

from utils.Target import Target


def test_target_encoding():
    target = Target()

    labels = np.array(["cat", "dog", "cat", "bird"])

    target.load_new_data(labels)

    assert len(target.data) == 4

    assert set(target.data) == {0, 1, 2}

    assert set(target.decrypt_map.keys()) == {"bird", "cat", "dog", }


def test_target_output_is_numeric():
    target = Target()

    target.load_new_data(np.array(["a", "b", "c"]))

    assert np.issubdtype(target.data.dtype, np.integer)
