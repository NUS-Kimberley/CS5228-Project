import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from src.model import light_model
import tensorflow as tf
from tensorflow import keras

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

def load_data(path = "./data/preprocessed_train_data.csv"):

    train_data = pd.read_csv(path)

    X = train_data.iloc[:,0:-1]
    X = np.array(X)
    y = None
    if 'prince' in train_data.columns:
        y = train_data["price"]
    else:
        y = train_data.iloc[:,-1]
    y = np.array(y)

    print(X.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_predict_data(path = "./data/preprocessed_test_data.csv"):
    test_data = pd.read_csv(path)
    X = np.array(test_data.iloc[:,0:])
    return X
    
def get_k_fold_valid(get_model):
    train_data = pd.read_csv("./data/preprocessed_train_data.csv")

    X = train_data.iloc[:,1:-1]
    X = np.array(X)
    y = train_data["price"]
    y = np.array(y)

    kfold = KFold(n_splits=5, random_state=2021,shuffle=True)

    rmse_score = []
    
    # K-fold evaluation
    for train, test in kfold.split(X, y):
        model = get_model(X[train], y[train], X[test], y[test], is_train=False)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35)
        history = model.fit(X[train], y[train], batch_size=64, epochs=400, validation_split=0.2, callbacks=[callback])
        result = model.evaluate(X[test], y[test])
        print(result)
        rmse_score.append(result[1])

    print(rmse_score)
    print(np.mean(np.array(rmse_score)))
