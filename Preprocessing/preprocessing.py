from sklearn.pipeline import Pipeline
from data_imputer import AirportDataImputer, WeatherDataImputer
from data_augmenter import AirportDataAugmenter, GeoDataAugmenter, TrainDataAugmenter
from data_encoder import TrainDataEncoder
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
