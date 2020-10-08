from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import math
from scipy import stats
from Utils.utils import DatetimeToTaxitime, distanceInKmBetweenCoordinates, ComputeQandN,date_transfo


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
        
        # Computing the moving average of the taxitime

        df['AOBT'] = pd.to_datetime(df['AOBT'])
        moving_avg = df[['AOBT', 'taxitime']]
        moving_avg = moving_avg.set_index('AOBT')

        #Display each minute within 2015 & 2018
        moving_avg = moving_avg.groupby(pd.Grouper(freq = "min")).mean()
        moving_avg = moving_avg.reset_index()

        #Compute the moving average of taxitime for each row, rolling 2 months before (ie 87840 minutes)
        moving_avg['moving_avg'] = moving_avg['taxitime'].rolling(window = 87840, min_periods = 1).mean()

        #Shift 1 because the rolling average takes into account the actual  row
        moving_avg['moving_avg'] = moving_avg['moving_avg'].shift(1)

        #Round the nearest integer
        moving_avg['moving_avg'] = moving_avg['moving_avg'].round(0)

        # For the first 2 months, replace the values by the global mean
        moving_avg['moving_avg'][0:87840] = moving_avg['taxitime'].mean()

        moving_avg = moving_avg.dropna()
        moving_avg = moving_avg.reset_index()

        #Combine the moving_avg table with our clean dataset
        df = pd.merge(df, moving_avg[['AOBT', 'moving_avg']] ,on = 'AOBT', how='left')

        return df
