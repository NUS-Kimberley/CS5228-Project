from src.utils import load_data,load_predict_data
from src.model import baseline_model
from keras.models import load_model
import pandas as pd
import numpy as np
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    if sys.argv[1] == "train":
        x_train, y_train, x_test, y_test = load_data()
        baseline_model(x_train, y_train, x_test, y_test)

    if sys.argv[1] == "predict":
        X_predict = load_predict_data()
        model = load_model("./models/baseline_model.h5")
        y_predict = model.predict(X_predict)
        y_predict = np.squeeze(y_predict)
        print(y_predict[0:10], y_predict.shape)
        result = pd.DataFrame({
        "Id": range(0, y_predict.shape[0]),
        "Predicted": y_predict})
        result.to_csv("./data/submission.csv",index=None)

    if sys.argv[1] == "check":
        x_train, y_train, x_test, y_test = load_data()
        model = load_model("./models/baseline_model.h5")
        print(model.predict(x_test)[0:15])
        print(y_test[0:15])

    if sys.argv[1] == "tree":
        x_train, y_train, x_test, y_test = load_data()
        tree = DecisionTreeRegressor()
        tree.fit(x_train, y_train)
        y_predict = tree.predict(x_test)
        print(mean_squared_error(y_test,y_predict, squared=False))
        print(y_predict[0:15])
        print(y_test[0:15])

    if sys.argv[1] == "forest":
        x_train, y_train, x_test, y_test = load_data("./data/preprocessed_forest_train_data.csv")
        regr = RandomForestRegressor(max_depth=80, random_state=2021)
        regr.fit(x_train, y_train)
        y_predict = regr.predict(x_test)
        print(mean_squared_error(y_test,y_predict, squared=False))
        print(y_predict[0:15])
        print(y_test[0:15])
        X_predict = load_predict_data("./data/preprocessed_forest_test_data.csv")
        y_result = regr.predict(X_predict)
        result = pd.DataFrame({
        "Id": range(0, y_result.shape[0]),
        "Predicted": y_result})
        result.to_csv("./data/forest_submission.csv",index=None)
        
    if sys.argv[1] == "zero":
        x_train, y_train, x_test, y_test = load_data("./data/preprocessed_zero_train_data.csv")
        baseline_model(x_train, y_train, x_test, y_test)

        X_predict = load_predict_data("./data/preprocessed_zero_test_data.csv")
        model = load_model("./models/baseline_model.h5")
        y_predict = model.predict(X_predict)
        y_predict = np.squeeze(y_predict)
        print(y_predict[0:10], y_predict.shape)
        result = pd.DataFrame({
        "Id": range(0, y_predict.shape[0]),
        "Predicted": y_predict})
        result.to_csv("./data/zero_submission.csv",index=None)


