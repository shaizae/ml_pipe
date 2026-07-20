import os
from itertools import combinations
from unittest.mock import patch

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from classificetin_files.Classificetion import Classification
from utils.utils import create_shared_numpy, KFoldsTrainData, cleanup_shared_memory


def test_set_initializes_data(classifier):
    assert classifier.fetchers is not None
    assert classifier.targets is not None
    assert len(classifier._train_data) == 2


def test_process_limit_valid():
    Classification.process_limit(1)

    assert Classification._process_limit == 1


def test_process_limit_cannot_exceed_cpu_count():
    Classification.process_limit(os.cpu_count() + 100)

    assert Classification._process_limit == os.cpu_count()


def test_process_limit_negative():
    with pytest.raises(ValueError):
        Classification.process_limit(-1)


@pytest.mark.parametrize("ratio", [-0.1, -1])
def test_train_test_split_negative_ratio(classifier, ratio):
    with pytest.raises(ValueError):
        classifier.train_test_split(ratio)


@pytest.mark.parametrize("ratio", [1.1, 2])
def test_train_test_split_ratio_too_large(classifier, ratio):
    with pytest.raises(ValueError):
        classifier.train_test_split(ratio)


def test_train_test_split_runs(classifier):
    results = classifier.train_test_split(0.8)
    assert len(results) == 2


def test_k_folds_invalid_splits_low(classifier):
    with pytest.raises(ValueError):
        classifier.k_folds(1)


def test_k_folds_invalid_splits_too_large(classifier):
    dataset_size = len(classifier.targets.data)

    with pytest.raises(ValueError):
        classifier.k_folds(dataset_size + 1)


def test_k_folds_runs(classifier):
    results = classifier.k_folds(3)
    assert len(results) == 2


def test_leave_one_out_runs(classifier):
    results = classifier.leave_one_out()

    assert len(results) == 2


def test_fetcher_selection_updates_train_data(classifier):
    number_of_features = [1, 2]
    original_len = len(classifier._train_data)
    classifier.fetcher_selection(algorithm=chi2, number_of_features=number_of_features)
    assert len(classifier._train_data) == original_len
    assert len(classifier._features_index) == len(number_of_features)
    seen = []
    for i in classifier._features_index:
        assert len(i) in number_of_features
        t = tuple(i)
        assert t not in seen
        seen.append(t)


def test_set_accepts_numpy_target(iris_data):
    X, y = iris_data

    cls = Classification()
    cls.set(X, y.to_numpy(), [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)], )

    assert cls.fetchers is not None
    assert cls.targets is not None
    assert len(cls._train_data) == 2


def test_use_more_fetchers_that_exist_in_fetcher_selection(classifier):
    classifier.fetcher_selection(algorithm=chi2, number_of_features=[10, 20])
    for i in classifier._features_index:
        assert len(i) == 4


def test_brut_force_features(classifier):
    original_models = len(classifier._features_index)
    n_features = classifier.fetchers.data.shape[1]
    expected_combinations = [combo for i in range(n_features) for combo in combinations(range(n_features), i)]
    classifier.brut_force_features()
    assert len(classifier._features_index) == (original_models * len(expected_combinations))
    produced = [tuple(td) for td in classifier._features_index]
    for _ in range(original_models):
        for combo in expected_combinations:
            assert produced.count(combo) == original_models


@cleanup_shared_memory
def test_train_test_split_generator(classifier):
    x = create_shared_numpy(classifier.fetchers.data, "x")
    y = create_shared_numpy(classifier.targets.data, "y")

    data = list(classifier._train_test_split_generator(0.8, x, y))

    assert len(data) == len(classifier._train_data)

    for td in data:
        assert td.train_index is not None
        assert td.test_index is not None


@cleanup_shared_memory
def test_k_fold_generator(classifier, iris_data):
    x, y = iris_data
    x = create_shared_numpy(x.values, "x")
    y = create_shared_numpy(y, "y")

    folds = list(classifier._k_fold_generator(kf=3, X_train=x, y_train=y, shuffle=False))

    assert len(folds) == len(classifier._train_data)

    for fold in folds:
        assert isinstance(fold, KFoldsTrainData)
        assert fold.number_of_folds == 3
        assert fold.shuffle is False

        assert fold.features.shape == x.shape
        assert fold.target.shape == y.shape


def test_train_test_split_exception(classifier):
    with patch(
            "classificetin_files.Classificetion.Pool",
            side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            classifier.train_test_split()


def test_k_folds_exception(classifier):
    with patch("classificetin_files.Classificetion.Pool", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            classifier.k_folds(3)


def test_fetcher_selection_exception(classifier):
    with patch("classificetin_files.Classificetion.Pool", side_effect=RuntimeError("boom"), ):
        with pytest.raises(RuntimeError, match="boom"):
            classifier.fetcher_selection(
                algorithm=chi2,
                number_of_features=[1, 2],
            )


def test_k_folds_no_shuffle(classifier):
    results = classifier.k_folds(3, shuffle=False)
    assert len(results) == 2
