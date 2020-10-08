from sklearn.pipeline import Pipeline
from Preprocessing.data_imputer import AirportDataImputer, WeatherDataImputer
from Preprocessing.data_augmenter import AirportDataAugmenter, GeoDataAugmenter, TrainDataAugmenter
from Preprocessing.data_encoder import TrainDataEncoder
import pandas as pd
# from my_standard_scaler import MyStandardScaler
# from my_one_hot_encoder import MyOneHotEncoder

class AirportPreprocessingPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
    			('DataImputer', AirportDataImputer()),
                ('DataAugmenter', AirportDataAugmenter()),
    			# ('MyStandardScaler', MyStandardScaler()),
    			# ('MyOneHotEncoder', MyOneHotEncoder())
	])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)

class GeoPreprocessingPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
    			# ('DataImputer', GeoDataImputer()),
                ('DataAugmenter', GeoDataAugmenter()),
    			# ('MyStandardScaler', MyStandardScaler()),
    			# ('MyOneHotEncoder', MyOneHotEncoder())
	])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)

class WeatherPreprocessingPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
    			('DataImputer', WeatherDataImputer())
                # ('DataAugmenter', WeatherDataAugmenter()),
    			# ('MyStandardScaler', MyStandardScaler()),
    			# ('MyOneHotEncoder', MyOneHotEncoder())
	])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)

class TrainPreprocessingPipeline:

    def __init__(self):
        self.pipeline = Pipeline([
    			#('DataImputer', TrainDataImputer()),
                ('DataAugmenter', TrainDataAugmenter()),
    			# ('MyStandardScaler', MyStandardScaler()),
    			('Encoder', TrainDataEncoder())
	])

    def fit(self, x):
        return self.pipeline.fit(x)

    def transform(self, x):
        return self.pipeline.transform(x)

def preprocess_datasets(airport_dataset_path, geo_dataset_path, weather_dataset_path, aircraft_dataset_path, excel = False):
    # 2. Loading the datasets and creating preprocessing pipelines
    print("Loading the datasets and creating pipelines...")
    if excel:
        df_airport_initial = pd.read_excel(airport_dataset_path)
        df_weather_initial = pd.read_excel(weather_dataset_path)

    else:    
        df_airport_initial = pd.read_csv(airport_dataset_path)
        df_weather_initial = pd.read_csv(weather_dataset_path)

    df_geographic_initial = pd.read_csv(geo_dataset_path, sep = ";") 
    df_aircraft = pd.read_csv(aircraft_dataset_path, sep = ";")

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
    df_initial = pd.merge(df_airport, df_geographic ,on='runway_stand',how='left')

    # Combining the training set with the weather dataset (the key is the datetime)
    df_initial = pd.merge(df_initial, df_weather ,on = 'AOBT_hourly', how='left')

    # Combining the training set with the aircraft characteristics dataset (the key is the datetime)
    df_initial = pd.merge(df_initial, df_aircraft ,on = 'aircraft_model', how='left')

    print('Combining datasets done')

    # 5. Cleaning the training dataset
    print("Cleaning the training dataset...")
    df = tpp.fit(df_initial)
    df = tpp.transform(df_initial)
    print('Cleaning done\n')

    return df
