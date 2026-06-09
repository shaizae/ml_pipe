from utils.Target import Target
import numpy as np

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




def test_pop_target_data():
    target = Target()
    target.load_new_data(np.array(["a", "b", "c", "d", "e", "f"]))

    popped = target.pop_index([1, 5, 3])
    assert np.array_equal(
        target._encoder.inverse_transform(popped), np.array(["b", "f", "d"]))

    assert np.array_equal(target.show_original_data, np.array(["a", "c", "e"]))
