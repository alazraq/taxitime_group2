from sklearn.ensemble import GradientBoostingRegressor

class GBMModel:

    def __init__(self):

        self.model = GradientBoostingRegressor(random_state=0, max_depth = 5,  n_estimators=150,learning_rate = 0.1, subsample = 0.8, verbose=True)
        

    def fit(self, x_train, y_train, x_test, y_test, early_stopping_rounds=10):
        features = ['Length__ft', 'Wake_Category', 'distance','precipIntensity', 'temperature', 'windBearing', 'cloudCover', 'visibility','precipAccumulation',  'N', 'Q','hour', 'month', 'quarter','off-peak_hour', 'i_clear-day','i_clear-night', 'i_cloudy', 'i_fog', 'i_partly-cloudy-day','i_partly-cloudy-night', 'i_rain', 'i_snow', 'i_wind','pt_rain', 'pt_snow', 'aircraft_taxitime', 'moving_avg']
        x_train = x_train[features]
        x_test = x_test[features]
        self.history = self.model.fit(x_train, y_train)

    def predict(self, x_test):
        pred = self.model.predict(x_test)
        return pred