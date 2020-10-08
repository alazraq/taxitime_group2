from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class TrainDataEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, df_train, y=None):
        df_train = df_train.join(pd.get_dummies(df_train['icon'], prefix='i'))

        df_train = df_train.join(pd.get_dummies(df_train['precipType'], prefix='pt'))

        #For the Wake Category, we are going to use Label Encoder as there is a linear combination between the wake and the taxitime
        df_train.Wake_Category.replace({'L': 1, 'M': 2, 'H': 3}, inplace=True)
        return df_train