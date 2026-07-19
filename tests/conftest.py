import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid

from utils.Fetchers import Fetchers
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from classificetin_files.Classificetion import Classification
from utils.utils import create_shared_numpy
import numpy as np

RANDOM_STATE = 42

N_ESTIMATORS = 10

SPLIT_TRAIN_SIZE = 100
SPLIT_TEST_SIZE = 50

K_FOLDS = 5
LEAVE_ONE_OUT_FOLDS = 150

SELECTED_FEATURES = 2
ALL_FEATURES = 10

FEATURE_INDEXES = [0, 1, 2, 3]

IRIS_FEATURE_COUNT = 4
IRIS_SAMPLES = 150

TEST_ARRAY_SHAPE = (5, 4)
TEST_ARRAY_SIZE = 20


@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

@pytest.fixture
def iris_data_sherd_memory():
    data = load_iris()
    X=create_shared_numpy(data.data, "fetchers")
    y=create_shared_numpy(data.target, "target")
    return X, y

@pytest.fixture
def classifier(iris_data):
    X, y = iris_data
    cls = Classification()
    cls.set(X, y, [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)])
    cls.process_limit(15)
    return cls


@pytest.fixture
def shared_array():
    arr = np.arange(20).reshape(5, 4)
    shm = create_shared_numpy(arr, f"test_{uuid.uuid4().hex}")
    yield shm, arr
    shm.unlink()


@pytest.fixture
def fetchers_with_data(iris_data):
    df, _ = iris_data
    f = Fetchers()
    f.load_new_data(df)
    return f


class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


@pytest.fixture
def logistic_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=500,
        random_state=RANDOM_STATE
    )


@pytest.fixture
def train_indexes():
    return list(range(SPLIT_TRAIN_SIZE))


@pytest.fixture
def test_indexes():
    return list(range(SPLIT_TRAIN_SIZE, SPLIT_TRAIN_SIZE + SPLIT_TEST_SIZE))
