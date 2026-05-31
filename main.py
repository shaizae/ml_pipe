import pandas as pd
from sklearn.datasets import  fetch_covtype
from Classificetion import Classification
def main():
    data = fetch_covtype()
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
    test_class=Classification()

    test_class.load(X,y)

if __name__ == '__main__':
    main()


