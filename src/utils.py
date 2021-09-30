import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path = "./data/preprocessed_train_data.csv"):

    train_data = pd.read_csv(path)

    X = train_data.iloc[:,1:-1]
    X = np.array(X)
    y = train_data["price"]
    y = np.array(y)

    print(X.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_predict_data(path = "./data/preprocessed_test_data.csv"):
    test_data = pd.read_csv(path,index=None)
    X = np.array(test_data)
    return X