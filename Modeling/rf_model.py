from sklearn.ensemble import RandomForestRegressor

class RFModel:

    def __init__(self):
        self.model = RandomForestRegressor(
         n_estimators=20, max_depth=13, max_features = 15)
        

    def fit(self, x_train, y_train, x_test, y_test, early_stopping_rounds=10):
        features = ['Length__ft', 'Wake_Category', 'distance', 'precipIntensity', 'temperature', 'windBearing', 'cloudCover', 'visibility','precipAccumulation',  'N', 'Q','hour', 'month', 'quarter', 'off-peak_hour', 'i_clear-day','i_clear-night', 'i_cloudy', 'i_fog', 'i_partly-cloudy-day','i_partly-cloudy-night', 'i_rain', 'i_snow', 'i_wind', 'pt_rain', 'pt_snow', 'aircraft_taxitime', 'moving_avg']
        x_train = x_train[features]
        x_test = x_test[features]
        self.history = self.model.fit(x_train, y_train)

    def predict(self, x_test):
        features = ['Length__ft', 'Wake_Category', 'distance', 'precipIntensity', 'temperature', 'windBearing', 'cloudCover', 'visibility','precipAccumulation',  'N', 'Q','hour', 'month', 'quarter', 'off-peak_hour', 'i_clear-day','i_clear-night', 'i_cloudy', 'i_fog', 'i_partly-cloudy-day','i_partly-cloudy-night', 'i_rain', 'i_wind', 'pt_rain', 'pt_snow', 'aircraft_taxitime']
        x_test = x_test[features]
        pred = self.model.predict(x_test)
        return pred