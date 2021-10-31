import numpy as np
import pandas as pd

def norm (x):
    return x - (x.sum()/np.count_nonzero(x))

def cal_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calculate_sim(car_array, item):
    sim = np.array([cal_sim(car_array[i], item) for i in range(0,len(car_array))])
    return sim


