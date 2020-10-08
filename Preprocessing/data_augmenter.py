from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import math
from scipy import stats
from Utils.utils import DatetimeToTaxitime, distanceInKmBetweenCoordinates, ComputeQandN,date_transfo, add_moving_avg


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

    def transform(self, df, y=None):
        # Drops the missing values for all the column where they correspond to less than 1% of the dataset
        subset_delete = ['Manufacturer', 'Engines', 'Wingspan__ft', 'Length__ft', 'Tail_Height__ft', 'Wheelbase__ft', 'Wake_Category', 'temperature', 'apparentTemperature', 'dewPoint','humidity','windSpeed', 'visibility', 'precipType', 'precipAccumulation', 'ozone']
        df = df.dropna(subset = subset_delete) 

        # For the remaining NAs value (weather data), 
        # Uses the method fillna assuming that the weather does not change as much from one hour to another
        df = df.fillna(method='ffill')

        # For some outliers, the ATOT is before the AOBT: there is a mistake so we delete them
        df = df[df["taxitime"] > 0]

        # Creates circular hours and several date-related columns
        df = date_transfo(df)
        
        #Delete useless columns
        df = df.drop(['flight_dt', 'ATOT', 'AOBT_hourly'], axis = 1)

         #Features per aircraft model
        avg_aircraft = df[['aircraft_model', 'taxitime']].groupby('aircraft_model').mean()

        avg_aircraft.rename(columns={'taxitime':'aircraft_taxitime'}, 
                        inplace=True)

        df = pd.merge(df, avg_aircraft ,on='aircraft_model',how='left')

        # df = add_moving_avg(df)

        return df
