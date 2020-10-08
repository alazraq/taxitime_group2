import numpy as np
import math
from datetime import timedelta
import time
import pandas as pd

# Compute distance between runways & stands in kilometers
def degreesToRadians(degrees):
    return degrees * np.pi / 180

def distanceInKmBetweenCoordinates(lat1, lon1, lat2, lon2):
    earthRadiusKm = 6371
    dLat = degreesToRadians(lat2-lat1)
    dLon = degreesToRadians(lon2-lon1)

    lat1 = degreesToRadians(lat1)
    lat2 = degreesToRadians(lat2)

    calculation = np.sin(dLat/2) * np.sin(dLat/2) + np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2) 
    distance = 2 * math.atan2(np.sqrt(calculation), np.sqrt(1-calculation))
    return earthRadiusKm * distance


# Computes the taxi-time
def DatetimeToTaxitime(datetime1, datetime2):
        difference = datetime2 - datetime1
        taxitime = difference.value / 6e10
        return taxitime

    
# Rounds to nearest hour by adding a timedelta hour if minute >= 30
def hour_rounder(t):
        return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
                +timedelta(hours=t.minute//30))

    
#Checks if a date is in the interval [date1,date2]
def CheckDateInInterval(date,date1,date2):
    if date >= date1 and date2 >= date:
        return 1
    else:
        return 0
    
# Computes Q and N
def ComputeQandN(df_airport):
    #Initializes N & Q list
    Q_list = []
    N_list = []

    #Creates one list of all AOBTs and one list of all ATOTs
    interval_list1 = list(df_airport['AOBT'])
    interval_list2 = list(df_airport['ATOT'])

    start_time = time.time()

    # Here we assume at most, there is as much airplanes in taxitime at the same time than the number of stands 
    window = len(df_airport['Stand'].unique())

    for row in range(len(df_airport)):
        #Setting the AOBT and ATOT of a given row
        AOBT = interval_list1[row]
        ATOT = interval_list2[row]
        
        #Defining the min and max rows in the dataframe this iteration will have to consider
        min_row = max(0, row - window)
        max_row = min(row + window, len(df_airport))
        
        short_list1 = interval_list1[min_row : max_row]
        short_list2 = interval_list2[min_row : max_row]
        
        #Creating a list of Booleans where there is 1 if for this iteration the row meets the condition for N
        N_boolean = map(lambda x, y : CheckDateInInterval(AOBT, x, y), short_list1, short_list2)
        #Computing the sum of the airplane satisfying the condition for N
        N_number = max(sum(list(N_boolean))-1, 0)
        N_list.append(N_number)

        #Creating a list of Booleans where there is 1 if for this iteration the row meets the condition for Q
        Q_boolean = map(lambda y : CheckDateInInterval(y, AOBT, ATOT), short_list2)
        #Computing the sum of the airplane satisfying the condition for Q
        Q_number = max(0, sum(list(Q_boolean))-1)
        Q_list.append(Q_number)
        
        if row % 50000 == 0:
            running_time = time.time() - start_time
            print("Row number: ", row, "/ Running time: " , running_time)
    return N_list, Q_list


# Transforms date into several columns and computes circular hours
def date_transfo(df):
    #Transform Date into several column
    pi = np.pi
    df['hour'] = df['AOBT'].dt.hour
    df['month'] = df['AOBT'].dt.month
    df['quarter'] = df['AOBT'].dt.quarter
    
    #Where average taxitime < 15 mins
    df['off-peak_hour'] = ((df['hour'] <4) + (df['hour'] >21 ))*1
    
    # Get circular hour
    df['hour_sin'] = np.sin(pi* df['hour'].astype(np.float64) /12)
    df['hour_cos'] = np.cos(pi* df['hour'].astype(np.float64) /12)
    df['month_sin'] = np.sin(pi* df['month'].astype(np.float64) /6)
    df['month_cos'] = np.cos(pi* df['month'].astype(np.float64) /6)
    df['quarter_sin'] = np.sin(pi* df['quarter'].astype(np.float64) /2)
    df['quarter_cos'] = np.cos(pi* df['quarter'].astype(np.float64) /2)

    return(df)

def add_moving_avg(df):
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

