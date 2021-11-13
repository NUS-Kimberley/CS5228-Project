import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def norm (x):
    return x - (x.sum()/np.count_nonzero(x))

def cal_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calculate_sim(car_array, item):
    sim = np.array([cal_sim(car_array[i], item) for i in range(0,len(car_array))])
    return sim

# Task2-track2: normalize numerical feature to create user profile
def normalize_numerical_features(df):
    # select numerical features (omit 'listing_id')
    numerical_features = df.dtypes[df.dtypes == 'float64'].index.tolist()
    
    df = df.copy()
    
    # normalize
    df[numerical_features] = df[numerical_features].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return df

# Task2-track2: encode categorical features to create user profile
def encode_categorical_features(df):
    # select categorical features
    categorical_features = df.dtypes[df.dtypes == 'object'].index.tolist()
    
    # one-hot encoding
    catogories_encoded = pd.get_dummies(df[categorical_features])
    df = df.join(catogories_encoded)
    df = df.drop(columns=categorical_features)
    return df

# Task2-track2: get user profile
def get_user_profile(listingid_score_dict, processed_df, non_zero_column_num):
    scores = np.array(list(listingid_score_dict.values()))
    scores_norm = (scores - scores.mean()).T
    rated_items = processed_df.set_index('listing_id').loc[list(listingid_score_dict.keys())].reset_index(inplace=False).drop(columns=['listing_id'])
    user_profile = scores_norm.dot(rated_items) / non_zero_column_num
    return user_profile

# Task2-track2: compute cosine similarity for all items
def calculate_cos_similar_for_all_items(user_profile, user_item_index, df):
    similarity_list = []
    user_profile = np.array(user_profile).reshape(1,user_profile.shape[0])
    unrated_items = df[~df['listing_id'].isin(user_item_index)]
    for index, row in unrated_items.iterrows():
        row_cleaned = np.array(row[1:]).reshape(1,user_profile.shape[1])
        simi = cosine_similarity(row_cleaned, user_profile)
        t = (simi[0][0], row['listing_id'])
        similarity_list.append(t)
    return similarity_list