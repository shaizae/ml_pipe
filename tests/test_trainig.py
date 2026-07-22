import numpy as np
import pytest
from sklearn.feature_selection import f_classif

from tests.conftest import FEATURE_INDEXES, K_FOLDS, LEAVE_ONE_OUT_FOLDS, RANDOM_STATE, IRIS_SAMPLES, SELECTED_FEATURES, \
    IRIS_FEATURE_COUNT, ALL_FEATURES, iris_data_sherd_memory
from results.ClassificationResults import ClassificationResults
from utils.multiprocess_functions import train_test_split_mc, train_k_folds_mc, features_selections
from utils.utils import TrainData, KFoldsTrainData, FeaturesSelectionsData, ValidationType, features_unpackage, \
    cleanup_shared_memory


def test_features_unpackage(shared_array):
    shm, original = shared_array

    indexes = [0, 2, 4]
    feature_indexes = [1, 3]

    result = features_unpackage(shm, indexes, feature_indexes)

    expected = original[indexes][..., feature_indexes]

    np.testing.assert_array_equal(result, expected)

@cleanup_shared_memory
def test_train_test_split_mc( logistic_model, train_indexes, test_indexes, iris_data_sherd_memory):
    features, target = iris_data_sherd_memory

    data = TrainData(model=logistic_model, features=features, target=target, train_index=train_indexes,
                     test_index=test_indexes, featuresIndex=FEATURE_INDEXES)

    result = train_test_split_mc(data)

    assert isinstance(result, ClassificationResults)

    assert result.validation == ValidationType.train_test_split

    assert result._pred_test is not None

@cleanup_shared_memory
@pytest.mark.parametrize("folds,expected_validation",
                         [(K_FOLDS, ValidationType.k_folds), (LEAVE_ONE_OUT_FOLDS, ValidationType.leave_one_out)])
def test_train_k_folds_mc(iris_data_sherd_memory, logistic_model, folds, expected_validation):
    features, target = iris_data_sherd_memory

    data = KFoldsTrainData(model=logistic_model, features=features, target=target, number_of_folds=folds, shuffle=True,
                           randon_state=RANDOM_STATE, featuresIndex=FEATURE_INDEXES)

    result = train_k_folds_mc(data)

    assert isinstance(result, ClassificationResults)

    assert result.validation == expected_validation

    assert result._pred_test is not None

    assert len(result._pred_test) == IRIS_SAMPLES

@cleanup_shared_memory
def test_features_selection(iris_data_sherd_memory):
    features, target = iris_data_sherd_memory

    data = FeaturesSelectionsData(features=features, target=target, algorithm=f_classif,
                                  number_of_features=SELECTED_FEATURES)

    result = features_selections(data)

    assert isinstance(result, np.ndarray)

    assert len(result) == SELECTED_FEATURES

    assert all(0 <= x < IRIS_FEATURE_COUNT for x in result)

@cleanup_shared_memory
def test_features_selection_when_request_all(iris_data_sherd_memory):
    features, target = iris_data_sherd_memory

    data = FeaturesSelectionsData(features=features, target=target, algorithm=f_classif,
                                  number_of_features=ALL_FEATURES)

    result = features_selections(data)

    assert result == list(range(IRIS_FEATURE_COUNT))
