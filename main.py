# 1. Loading the necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from preprocessing import AirportPreprocessingPipeline, GeoPreprocessingPipeline, WeatherPreprocessingPipeline, TrainPreprocessingPipeline
from scipy import stats
import os.path
from xgb_model import XGBModel
import argparse

# Parsing the flag dataset_loaded, if the flag is up skip the preprocessing
parser = argparse.ArgumentParser(description='Parser for main.')
parser.add_argument('--dataset_loaded', action='store_true', help='Flag used to skip the preprocessing if the preprocessed dataset is already available.')
args = parser.parse_args()
dataset_loaded = args.dataset_loaded

if not(dataset_loaded):
    # 2. Loading the datasets and creating preprocessing pipelines
    print("Loading the datasets and creating pipelines...")
    df_airport_initial = pd.read_csv("Raw_data/training_set_airport_data.csv")
    df_geographic_initial = pd.read_csv("Prepared_data/new_geographic_data.csv", sep = ";")
    df_weather_initial = pd.read_csv("Raw_data/Weather_data/weather_data_train_set.csv")
    df_aircraft = pd.read_csv("Prepared_data/ACchar.csv", sep = ";")

    app = AirportPreprocessingPipeline()
    gpp = GeoPreprocessingPipeline()
    wpp = WeatherPreprocessingPipeline()
    tpp = TrainPreprocessingPipeline()
    print("Loading datasets done")

    # 3. Preprocessing
    # Each dataset is preocessed by the pipelines' fit and transform functions
    print("Preprocessing the datasets...") 
    df_airport = app.fit(df_airport_initial)
    df_airport = app.transform(df_airport_initial)
    print("- Airport data preprocessed")
    df_geographic = gpp.fit(df_geographic_initial)
    df_geographic = gpp.transform(df_geographic_initial)
    print("- Geo data preprocessed")
    df_weather = wpp.fit(df_weather_initial)
    df_weather = wpp.transform(df_weather_initial)
    print("- Weather data preprocessed")
    print('Preprocessing done')

    # 4.Combining datasets
    print("Combining the datasets into a single training dataset...")
    # Combining the training set with the geographic dataset (the key is the runway & the stand)
    df_train_initial = pd.merge(df_airport, df_geographic ,on='runway_stand',how='left')

    # Combining the training set with the weather dataset (the key is the datetime)
    df_train_initial = pd.merge(df_train_initial, df_weather ,on = 'AOBT_hourly', how='left')

    # Combining the training set with the aircraft characteristics dataset (the key is the datetime)
    df_train_initial = pd.merge(df_train_initial, df_aircraft ,on = 'aircraft_model', how='left')

    print('Combining datasets done')

    # 5. Cleaning the training dataset
    print("Cleaning the training dataset...")
    df_train = tpp.fit(df_train_initial)
    df_train = tpp.transform(df_train_initial)
    print('Cleaning done\n')

    # Saving the dataset TAKES A LOT OF TIME
    # df_train_initial.to_csv("Preprocessed_airport_data.csv", index= False)

else:
    dataset_in_path = os.path.isfile('clean_training_set_vf.csv')
    if dataset_in_path:
        print("The preprocessed dataset is loaded, skipping the preprocessing")
    else:
        raise ValueError("Cannot find the clean dataset...")
    df_train = pd.read_csv("clean_training_set_vf.csv")

print(f"Training dataset shape is {df_train.shape}")
print('\n')
print(df_train.head())

# 6. Splitting into training and validation datasets
print("Splitting into training and validation datasets...")
df_y = df_train['taxitime']
df_X = df_train.drop(['taxitime'], axis=1)

# Split test & train randomly
X_train, X_test, y_train, y_test = train_test_split(
     df_X, df_y, test_size=0.2, random_state=42)

# 7. Models definition

# XGBoost
model_xgb = XGBModel()
model_xgb.fit(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
y_predict = model_xgb.predict(test)
print(y_predict[0:5])

