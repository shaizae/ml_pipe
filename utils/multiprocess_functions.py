import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold

from utils.Results import Results
from utils.utils import TrainData, KFoldsTrainData, FeaturesSelectionsData, force_gc

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)


@force_gc
def train_test_split_mc(train_data: TrainData):
    features, target = train_data.get_train()
    train_data.fit(features, target)
    result = Results(train_data.model)
    features, target = train_data.get_test()
    result.set_test(features, target)
    result.predict()
    return result


@force_gc
def train_k_folds_mc(train_data: KFoldsTrainData):
    kf = KFold(n_splits=train_data.number_of_folds, shuffle=train_data.shuffle)
    final_result: Results = None
    for train_index, test_index in kf.split(train_data.features.array):
        fold_data = train_data.copy()
        fold_data.set_indexes(train_index, test_index)
        features, target = train_data.get_train()
        result = Results(fold_data.fit(features, target))
        features, target = train_data.get_test()
        result.set_test(features, target)
        if final_result is None:
            final_result = result
        final_result.append_results(result)
    final_result.predict()
    return final_result


@force_gc
def features_selections(features_selections_data: FeaturesSelectionsData):
    features = features_selections_data.features.array
    target = features_selections_data.target.array
    if features_selections_data.number_of_features >= features.shape[1]:
        return list(range(features.shape[1]))
    algorithm = features_selections_data.algorithm
    select = SelectKBest(algorithm, k=features_selections_data.number_of_features)
    select.fit(features, target)
    return np.where(select.get_support())[0]
