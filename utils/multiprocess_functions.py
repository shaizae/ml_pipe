import warnings

import numpy as np
from sklearn.feature_selection import SelectKBest

from utils.Results import Results
from utils.utils import TrainData, FeaturesSelectionsData, SharedMemory, force_gc
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="sklearn"
)

@force_gc
def features_unpackage(shared_memory: SharedMemory, index: list[int], features_index: list[int]):
    return shared_memory.array[index][..., features_index]



def train(train_data: TrainData):
    if train_data.print:
        print(f"train start whit model: {train_data.name}")
    train_data(features_unpackage(train_data.features, train_data.train_index, train_data.featuresIndex),
               train_data.target.array[train_data.train_index])
    if train_data.print:
        print(f"train end whit model: {train_data.name}")

    result = Results(train_data.model)
    result.set_test(features_unpackage(train_data.features, train_data.test_index, train_data.featuresIndex),
                    train_data.target.array[train_data.test_index])
    if train_data.print:
        print(f"predicting results end whit model: {train_data.name}")
    return result


def features_selections(features_selections_data: FeaturesSelectionsData):
    features = features_selections_data.features.array
    target = features_selections_data.target.array
    if features_selections_data.number_of_features >= features.shape[1]:
        return list(range(features.shape[1]))
    algorithm = features_selections_data.algorithm
    select = SelectKBest(algorithm, k=features_selections_data.number_of_features)
    select.fit(features, target)
    return np.where(select.get_support())[0]
