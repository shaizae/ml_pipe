import pandas as pd
from sklearn.datasets import load_iris as DB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

from external_classes.Classificetion import Classification
from utils.utils import FilteringCriteria


def main():
    train_test_split()
    k_folds()
    leave_one_out()


def train_test_split():
    data = DB()
    feature_names = data.feature_names

    X = pd.DataFrame(data.data, columns=feature_names)
    target_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    y = pd.Series(data.target).map(target_names)
    test_class = Classification()

    test_class.set(X, y, [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC()])
    test_class.fetchers.min_max_scaler()
    test_class.brut_force_features()
    test_class.train_test_split()
    filtered_results = test_class.filter_by(FilteringCriteria.accuracy)
    for i in tqdm(filtered_results):
        i.save_results(r".\tests\pdf")


def k_folds():
    data = DB()
    feature_names = data.feature_names

    X = pd.DataFrame(data.data, columns=feature_names)
    target_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    y = pd.Series(data.target).map(target_names)
    test_class = Classification()

    test_class.set(X, y, [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC()])
    test_class.fetchers.min_max_scaler()
    test_class.fetcher_selection(chi2, [1, 2, 3])
    test_class.k_folds()
    filtered_results = test_class.filter_by(FilteringCriteria.precision)
    for i in tqdm(filtered_results):
        i.save_results(r".\tests\pdf")


def leave_one_out():
    data = DB()
    feature_names = data.feature_names

    X = pd.DataFrame(data.data, columns=feature_names)
    target_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"
    }
    y = pd.Series(data.target).map(target_names)
    test_class = Classification()

    test_class.set(X, y, [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC()])
    test_class.fetchers.standard_scaler()
    test_class.fetchers.polynomial_features(degree=3, include_bias=True)
    test_class.set_hyper_parameter_brut_force(
        {"n_estimators": [50, 100, 120], "min_samples_split": [2, 3, 6],"C":[1,2,3]})
    test_class.leave_one_out()
    filtered_results = test_class.filter_by(FilteringCriteria.f1)
    for i in tqdm(filtered_results):
        i.save_results(r".\tests\pdf")


if __name__ == '__main__':
    main()
