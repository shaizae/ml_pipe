import os

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from classificetin_files.Classificetion import Classification


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
    classifier.process_limit(2)
    classifier.fetcher_selection(algorithm=chi2, number_of_features=number_of_features)
    assert len(classifier._train_data) == original_len * 2
    for filtered,expected in zip(classifier._train_data,number_of_features):
        assert len(filtered.featuresIndex) == expected



def test_set_accepts_numpy_target(iris_data):
    X, y = iris_data

    cls = Classification()
    cls.set(X, y.to_numpy(), [RandomForestClassifier(n_estimators=10, random_state=42), SVC(probability=True)], )

    assert cls.fetchers is not None
    assert cls.targets is not None
    assert len(cls._train_data) == 2

def test_use_more_fetchers_that_exist_in_fetcher_selection(classifier):
    classifier.fetcher_selection(algorithm=chi2, number_of_features=[10, 20])
    for i in classifier._train_data:
        assert len(i.featuresIndex) ==4



