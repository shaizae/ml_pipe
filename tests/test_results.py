import numpy as np

from utils.Results import Results


class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


def test_append_results():
    r1 = Results(DummyModel())
    r2 = Results(DummyModel())

    r1.features_test = np.array([[1, 2]])
    r1.target_test = np.array([0])
    r1._pred_test = np.array([0])
    r1._predict_score = np.array([[0.5, 0.5]])

    r2.features_test = np.array([[3, 4]])
    r2.target_test = np.array([1])
    r2._pred_test = np.array([1])
    r2._predict_score = np.array([[0.2, 0.8]])

    r1.append_results(r2)

    assert len(r1.target_test) == 2

    assert np.array_equal(r1.target_test, np.array([0, 1]))

    assert r1.features_test.shape == (2, 2)

    assert r1._predict_score.shape == (2, 2)


def test_name_property():
    result = Results(DummyModel())

    assert result.name == "DummyModel"
