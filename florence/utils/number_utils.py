import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

class NumberUtils:
    @staticmethod
    def normalize_data(df):
        normalize_columns = set([
            "imbalance_size",
            "matched_size",
            "bid_size",
            "ask_size",
        ])
        normalize_columns = list(normalize_columns.intersection(set(df.columns)))

        scaler = preprocessing.StandardScaler()
        scaler.fit(df[normalize_columns])
        df[normalize_columns] = scaler.transform(df[normalize_columns])