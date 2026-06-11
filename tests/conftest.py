import sys
import os


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

@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def classifier(iris_data):
    X, y = iris_data
    cls = Classification()
    cls.set(X, y, [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)])
    return cls


@pytest.fixture
def shared_array():
    arr = np.arange(20).reshape(5, 4)
    shm = create_shared_numpy(        arr,        f"test_{uuid.uuid4().hex}"    )
    yield shm, arr
    shm.unlink()


@pytest.fixture
def fetchers_with_data(iris_data):
    df, _ = iris_data
    f = Fetchers()
    f.load_new_data(df)
    return f
