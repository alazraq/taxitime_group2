from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def rmse(y_actual, y_predict):
    rmse = sqrt(mean_squared_error(y_actual, y_predict))
    return rmse

def accuracy(y_actual, y_predict):
    length = len(y_actual)
    accuracy_1 = sum(((np.abs(y_actual) - np.abs(y_predict)) <= 1) *1) / length
    accuracy_2 = sum(((np.abs(y_actual) - np.abs(y_predict)) <= 2) *1) / length
    accuracy_3 = sum(((np.abs(y_actual) - np.abs(y_predict)) <= 3) *1) / length
    accuracy_5 = sum(((np.abs(y_actual) - np.abs(y_predict)) <= 5) *1) / length
    accuracy_10 = sum(((np.abs(y_actual) - np.abs(y_predict)) <= 10) *1) / length
    
    return accuracy_1, accuracy_2, accuracy_3, accuracy_5, accuracy_10