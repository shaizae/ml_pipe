import os

import pandas as pd
from sklearn.datasets import load_iris as DB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from classificetin_files.Classificetion import Classification


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

    test_class.set(X, y, [RandomForestClassifier(), AdaBoostClassifier(), DecisionTreeClassifier(), SVC()])
    test_class.fetchers.standard_scaler()
    # test_class.fetcher_selection(chi2,list(range(5,30,5)))
    res = test_class.leave_one_out()
    for i in res:
        i.save_pdf_report(r"C:\python_projrcts\ml_pipe\tests\pdf")


if __name__ == '__main__':
    main()
