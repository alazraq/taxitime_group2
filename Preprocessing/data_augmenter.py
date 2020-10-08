from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import math
from scipy import stats
from utils import DatetimeToTaxitime, distanceInKmBetweenCoordinates, ComputeQandN,date_transfo


class AirportDataAugmenter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x['taxitime'] = x.apply(lambda l: DatetimeToTaxitime(l.AOBT, l.ATOT), axis =1)
        x = x[x["taxitime"] > 0]
        x['runway_stand'] = x['Runway'] + x['Stand']
        N_list, Q_list = ComputeQandN(x)
        x['N'] = N_list        
        x['Q'] = Q_list
        return x

class GeoDataAugmenter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        # Compute the distance in the aircraft dataframe
        x['distance'] = x.apply(lambda x: distanceInKmBetweenCoordinates(x.Lat_runway, x.Lng_runway, x.Lat_stand, x.Lng_stand), axis =1)
        x['log_distance'] = np.log(x['distance'])
        x = x[['runway_stand', 'distance', 'log_distance']]
        return x

class TrainDataAugmenter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, df_train, y=None):
        # Drops the missing values for all the column where they correspond to less than 1% of the dataset
        subset_delete = ['Manufacturer', 'Engines', 'Wingspan__ft', 'Length__ft', 'Tail_Height__ft', 'Wheelbase__ft', 'Wake_Category', 'temperature', 'apparentTemperature', 'dewPoint','humidity','windSpeed', 'visibility', 'precipType', 'precipAccumulation', 'ozone']
        df_train = df_train.dropna(subset = subset_delete) 

        # For the remaining NAs value (weather data), 
        # Uses the method fillna assuming that the weather does not change as much from one hour to another
        df_train = df_train.fillna(method='ffill')

        # For some outliers, the ATOT is before the AOBT: there is a mistake so we delete them
        df_train = df_train[df_train["taxitime"] > 0]


        # For the others, we use Z-score strategy and delete points that fall outside of 3 standard deviations 
        z_score = np.abs(stats.zscore(df_train.taxitime))
        threshold = 3
        df_train = df_train[(z_score< threshold)]

        # Creates circular hours and several date-related columns
        df_train = date_transfo(df_train)
        
        #Delete useless columns
        df_train = df_train.drop(['flight_dt', 'ATOT', 'AOBT_hourly'], axis = 1)

         #Features per manufacturer
        avg_manuf = df_train[['Manufacturer', 'taxitime']].groupby('Manufacturer').mean()

        avg_manuf.rename(columns={'taxitime':'avg_manuf'}, 
                        inplace=True)

        df_train = pd.merge(df_train, avg_manuf ,on='Manufacturer',how='left')
        
        return df_train
