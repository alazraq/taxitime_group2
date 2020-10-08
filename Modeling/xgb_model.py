import xgboost as xgb

class XGBModel:

    def __init__(self, booster='gbtree', colsample_bytree=0.8, 
            importance_type='gain', learning_rate=0.1, max_depth=6, 
            min_child_weight=1, n_estimators=25, objective='reg:squarederror', 
            random_state=0, subsample=0.8, verbosity=0):
        self.model = xgb.XGBRegressor( 
            booster=booster, 
            colsample_bytree=colsample_bytree, 
            importance_type=importance_type, 
            learning_rate=learning_rate, 
            max_depth=max_depth, 
            min_child_weight=min_child_weight, 
            n_estimators=n_estimators, 
            objective=objective, 
            random_state=random_state, 
            subsample=subsample, 
            verbosity=verbosity)
        

    def fit(self, x_train, y_train, x_test, y_test, early_stopping_rounds=10):
        features= ['Engines','Wingspan__ft', 'Length__ft', 'Tail_Height__ft', 'Wheelbase__ft','Wake_Category', 'distance','precipIntensity', 'precipProbability', 'temperature','apparentTemperature', 'dewPoint', 'humidity', 'pressure', 'windSpeed','windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility', 'precipAccumulation', 'ozone', 'log_distance', 'N', 'Q','hour', 'month', 'quarter', 'off-peak_hour', 'hour_sin', 'hour_cos','month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'i_clear-day','i_clear-night', 'i_cloudy', 'i_fog', 'i_partly-cloudy-day','i_partly-cloudy-night', 'i_rain', 'i_snow', 'i_wind', 'pt_None','pt_rain', 'pt_snow', 'avg_manuf']
        x_train = x_train[features]
        x_test = x_test[features]
        self.history = self.model.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        early_stopping_rounds=early_stopping_rounds,
       verbose=False)

    def predict(self, x_test):
        pred = self.model.predict(x_test)
        return pred