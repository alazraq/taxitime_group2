# 1. Importing the necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from Preprocessing.preprocessing import AirportPreprocessingPipeline, GeoPreprocessingPipeline, WeatherPreprocessingPipeline, TrainPreprocessingPipeline, preprocess_datasets
from scipy import stats
import os.path
from Modeling.gbm_model import GBMModel
import argparse
from Utils.metrics import rmse, accuracy

# Parsing the flag dataset_loaded, if the flag is up skip the preprocessing
parser = argparse.ArgumentParser(description='Parser for main.')
parser.add_argument('--dataset_loaded', action='store_true', help='Flag used to skip the preprocessing if the preprocessed dataset is already available.')
args = parser.parse_args()
dataset_loaded = args.dataset_loaded

if not(dataset_loaded):
    airport_dataset_path = "Raw_data/training_set_airport_data.csv"
    geo_dataset_path = "Prepared_data/new_geographic_data.csv"
    weather_dataset_path = "Raw_data/Weather_data/weather_data_train_set.csv"
    aircraft_dataset_path = "Prepared_data/ACchar.csv"
    df_train = preprocess_datasets(airport_dataset_path, geo_dataset_path, weather_dataset_path, aircraft_dataset_path)

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
print("Successfully split into training and validation datasets.")
# 7. Models definition

# LightGBM
print("Fitting a LightGBM model...")
model_gbm =  GBMModel()
model_gbm.fit(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
print("Fitting LightGBM model done.")
# 8. Prediction

## Preparing the test set
airport_test_dataset_path = "Test_data/test_set_airport_data.xlsx"
geo_test_dataset_path = "Prepared_data/new_geographic_data.csv"
weather_test_dataset_path = "Test_data/test_set_weather_data.xlsx"
aircraft_test_dataset_path = "Prepared_data/ACchar.csv"
df_test = preprocess_datasets(airport_test_dataset_path, geo_test_dataset_path, weather_test_dataset_path, aircraft_test_dataset_path, excel = True)
df_test = df_test.drop(['aircraft_taxitime_x'], axis = 1)
df_test = df_test.drop(['aircraft_taxitime_y'], axis = 1)
df_test['aircraft_taxitime'] = df_test['aircraft_taxitime'].fillna(clean_test['taxitime'].mean())
## Predicting taxitime
y_predict = model_xgb.predict(df_test)
print(y_predict[0:5])