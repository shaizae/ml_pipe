import pytest


def test_set(classifier):
    assert classifier.fetchers is not None
    assert classifier.targets is not None

    assert len(classifier._train_data) == 2


def test_train_test_split_runs(classifier):
    results = classifier.train_test_split(
        ratio=0.2
    )

    assert len(results) == 2

    for result in results:
        assert result.target_test is not None
        assert result.features_test is not None


def test_k_folds_runs(classifier):
    results = classifier.k_folds(
        n_splits=3
    )

    assert len(results) == 2

    for result in results:
        assert result.target_test is not None
        assert result._pred_test is not None


def test_invalid_ratio_negative(classifier):
    with pytest.raises(ValueError):
        classifier.train_test_split(
            ratio=-1
        )


def test_invalid_ratio_large(classifier):
    with pytest.raises(ValueError):
        classifier.train_test_split(
            ratio=2
        )


def test_invalid_kfold_small(classifier):
    with pytest.raises(ValueError):
        classifier.k_folds(
            n_splits=1
        )


def test_invalid_kfold_large(classifier):
    with pytest.raises(ValueError):
        classifier.k_folds(
            n_splits=100000
        )