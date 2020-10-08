import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import hour_rounder


class AirportDataImputer(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.X = None

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # Rename the columns
        x.rename(columns={
            'Flight Datetime':'flight_dt',
            'Aircraft Model':'aircraft_model'}, 
        inplace=True)
        x['flight_dt'] =  pd.to_datetime(x['flight_dt'], format='%m/%d/%Y %H:%M')
        x['AOBT'] =  pd.to_datetime(x['AOBT'], format='%m/%d/%Y %H:%M')
        x['ATOT'] =  pd.to_datetime(x['ATOT'], format='%m/%d/%Y %H:%M')
        x['AOBT_hourly'] = x.apply(lambda l: hour_rounder(l.AOBT), axis =1)
        # x['AOBT_hourly'] = x.apply(lambda l: hour_rounder(pd.to_datetime(l[2], format='%m/%d/%Y %H:%M')), axis =1, raw= True)
        x = x.drop_duplicates()
        return x

class WeatherDataImputer(BaseEstimator, TransformerMixin):  

    def __init__(self):
        self.X = None

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x['time_hourly'] =  pd.to_datetime(x['time_hourly'], format='%m/%d/%Y %H:%M')
        x.rename(columns={'time_hourly':'AOBT_hourly'}, 
                        inplace=True)
        x = x.drop_duplicates()
        return x