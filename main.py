import pandas as pd
from sklearn.datasets import load_iris as DB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from Classificetion import Classification


def main():
    data = DB()
    feature_names = data.feature_names

    # Create DataFrame for features
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

    test_class.set(X, y, [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(),SVC()])
    res=test_class.train_test_split(0.6)
    for i in res:
        i.plot_confusion_matrix()
        i.plot_roc_curve()


if __name__ == '__main__':
    main()
