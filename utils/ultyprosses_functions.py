import numpy as np
from sklearn.feature_selection import SelectKBest

from utils.Results import Results
from utils.utils import TrainData, FeaturesSelectionsData


def train(train_data: TrainData):
    if train_data.print:
        print(f"train start whit model: {train_data.name}")
    train_features = train_data.features.array[train_data.train_index][:, train_data.featuresIndex]
    train_target = train_data.target.array[train_data.train_index]
    train_data(train_features, train_target)
    if train_data.print:
        print(f"train end whit model: {train_data.name}")
    result = Results(train_data.model)
    test_features = train_data.features.array[train_data.test_index][:, train_data.featuresIndex]
    test_target = train_data.target.array[train_data.test_index]
    result.set_test(test_features, test_target)
    if train_data.print:
        print(f"predicting results end whit model: {train_data.name}")
    return result


def features_selections(features_selections_data: FeaturesSelectionsData):
    features = features_selections_data.features.array
    target = features_selections_data.target.array
    algorithm = features_selections_data.algorithm
    select = SelectKBest(algorithm, k=features_selections_data.number_of_features)
    select.fit(features, target)
    return np.where(select.get_support())[0]
