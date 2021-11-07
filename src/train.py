from src.utils import load_data,load_predict_data,get_k_fold_valid,tree_k_fold_valid
from src.model import baseline_model, light_model, get_emb_model
from keras.models import load_model
import pandas as pd
import numpy as np
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

def train_and_get_deep_learning_model(model_type):
    x_train, y_train, x_test, y_test = load_data()
    if model_type == "baseline":
        model = baseline_model(x_train, y_train, x_test, y_test, is_train=True)
    elif model_type == "emb":
        model = get_emb_model(x_train, y_train, x_test, y_test, is_train=True)
    return model

def predict(model):
    X_predict = load_predict_data()
    y_predict = model.predict(X_predict)
    y_predict = np.squeeze(y_predict)
    print(y_predict[0:10], y_predict.shape)
    result = pd.DataFrame({
    "Id": range(0, y_predict.shape[0]),
    "Predicted": y_predict})
    result.to_csv("./data/submission.csv",index=None) 

def check_result():
    _, _, x_test, y_test = load_data()
    model = load_model("./models/baseline_model.h5")
    print(model.predict(x_test)[0:15])
    print(y_test[0:15])

def train_and_get_tree_model():
    x_train, y_train, x_test, y_test = load_data()
    tree = DecisionTreeRegressor()
    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    print(mean_squared_error(y_test,y_predict, squared=False))
    print(y_predict[0:15])
    print(y_test[0:15])
    return tree

def train_and_get_forest():
    x_train, y_train, x_test, y_test = load_data()
    regr = RandomForestRegressor(n_estimators = 200, random_state=2021)
    regr.fit(x_train, y_train)
    y_predict = regr.predict(x_test)
    print(mean_squared_error(y_test,y_predict, squared=False))
    print(y_predict[0:15])
    print(y_test[0:15])
    X_predict = load_predict_data()
    y_result = regr.predict(X_predict)
    result = pd.DataFrame({
    "Id": range(0, y_result.shape[0]),
    "Predicted": y_result})
    result.to_csv("./data/forest_submission.csv",index=None)
    return regr

def train_and_get_gbr():
    regr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, max_features='sqrt', min_samples_leaf=16, min_samples_split=8, random_state=2021) 
    x_train, y_train, x_test, y_test = load_data("./data/preprocessed_tree_train_data.csv")
    regr.fit(x_train, y_train)
    y_predict = regr.predict(x_test)
    print(mean_squared_error(y_test,y_predict, squared=False))
    print(y_predict[0:15])
    print(y_test[0:15])
    return regr


def k_fold(model):
    if model == "light":
        get_k_fold_valid(light_model)
    if model == "baseline":
        get_k_fold_valid(baseline_model)
    if model == "emb":
        get_k_fold_valid(get_emb_model)
    if model == "gbr":
        tree_k_fold_valid()

def get_blend_model(dl_model = None):
        model = None

        _, y_train, x_test, y_test = load_data()
        x_train_tree, _ , x_test_tree, _ = load_data("./data/preprocessed_tree_train_data.csv")

        if dl_model != None:
            model = dl_model
        else:
            model = load_model("./models/baseline_model.h5")

        dl_predict = np.squeeze(model.predict(x_test))
        # print(model.summary())
        
        gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, max_features='sqrt', min_samples_leaf=15,                                              min_samples_split=10, random_state = 2021) 
        gbr.fit(x_train_tree, y_train)
        gbr_predict = gbr.predict(x_test_tree)
        
        forest = RandomForestRegressor(n_estimators = 200, random_state=2021)
        forest.fit(x_train_tree, y_train)
        forest_predict = forest.predict(x_test_tree)
        
        y_predict = 0.3 * gbr_predict + 0.3 * forest_predict + 0.4 * dl_predict
        
        blend_score = mean_squared_error(y_test, y_predict, squared=False)
        
        gbr_score = mean_squared_error(y_test, gbr_predict, squared=False)
        forest_score = mean_squared_error(y_test, forest_predict, squared=False)
        dl_score = mean_squared_error(y_test, dl_predict, squared=False)
        
        print(gbr_score, forest_score, dl_score, blend_score)
        
        print(y_predict[0:10], y_test[0:10])
        print(gbr_predict[0:10])
        print(forest_predict[0:10])
        print(dl_predict[0:10])
        
        X_predict = load_predict_data()
        X_predict_tree = load_predict_data("./data/preprocessed_tree_test_data.csv")
        y_result = 0.3 * gbr.predict(X_predict_tree) + 0.3 * forest.predict(X_predict_tree) + 0.4 * np.squeeze(model.predict(X_predict))
        result = pd.DataFrame({
        "Id": range(0, y_result.shape[0]),
        "Predicted": y_result})
        result.to_csv("./data/blend_submission.csv",index=None)


if __name__ == '__main__':
    if sys.argv[1] == "train":
        model = train_and_get_deep_learning_model()
        predict(model)

    if sys.argv[1] == "predict":
        model = get_emb_model()
        model.load_weights("./models/emb_model")
        predict(model)
        
    if sys.argv[1] == "check":
        check_result()

    if sys.argv[1] == "tree":
        train_and_get_tree_model()

    if sys.argv[1] == "forest":
        train_and_get_forest()
  
    if sys.argv[1] == "gbr":
        train_and_get_gbr()
        
    if sys.argv[1] == "blend":
        get_blend_model()