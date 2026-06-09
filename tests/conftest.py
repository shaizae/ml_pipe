import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uuid

import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from classificetin_files.Classificetion import Classification
from utils.utils import create_shared_numpy


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
    import numpy as np

    arr = np.arange(20).reshape(5, 4)

    shm = create_shared_numpy(
        arr,
        f"test_{uuid.uuid4().hex}"
    )

    yield shm, arr

    shm.unlink()
