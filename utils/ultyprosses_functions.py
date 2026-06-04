import numpy as np

from Results import Results
from utils.utils import TrainData, FeaturesSelectionsData


def train(train_data: TrainData):
    if train_data.print:
        print(f"train start whit model: {train_data.name}")
    features = train_data.features.array
    target = train_data.target.array
    train_features = features[train_data.train_index, ...]
    train_target = target[train_data.train_index]
    train_data(train_features, train_target)
    if train_data.print:
        print(f"train end whit model: {train_data.name}")
    result = Results(train_data.model)
    test_features = features[train_data.test_index, ...]
    test_target = target[train_data.test_index]
    result.set_test(test_features, test_target)
    result.predict()
    if train_data.print:
        print(f"predicting results end whit model: {train_data.name}")
    return result


def features_selections(features_selections_data: FeaturesSelectionsData, ):
    features = features_selections_data.features.array
    target = features_selections_data.target.array
    algorithm = features_selections_data.algorithm
    algorithm.fit(features, target)
    return np.where(algorithm.get_support())[0]
