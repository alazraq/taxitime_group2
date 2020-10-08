from lightgbm import LGBMRegressor

class GBMModel:

    def __init__(self):

        self.model = LGBMRegressor(boosting_type = 'gbdt', learning_rate = 0.2, max_depth = -1, num_leaves = 100, n_estimators = 100, n_jobs = -1, subsample = 1, min_child_samples = 1, max_bin = 400, feature_fraction = 1)
        

    def fit(self, x_train, y_train, x_test, y_test, early_stopping_rounds=10):
        features = ['Length__ft', 'Wake_Category', 'distance', 'precipIntensity', 'temperature', 'windBearing', 'cloudCover', 'visibility', 'precipAccumulation',  'N', 'Q','hour', 'month', 'quarter', 'off-peak_hour', 'i_clear-day','i_clear-night', 'i_cloudy', 'i_fog', 'i_partly-cloudy-day','i_partly-cloudy-night', 'i_rain', 'i_snow', 'i_wind',   'pt_rain', 'pt_snow', 'aircraft_taxitime', 'moving_avg']
        x_train = x_train[features]
        x_test = x_test[features]
        self.history = self.model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],early_stopping_rounds=10, verbose=False)

    def predict(self, x_test):
        pred = self.model.predict(x_test)
        return pred