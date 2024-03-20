import numpy as np
import pandas as pd
from sklearn import preprocessing

from data_preprocessor.data_preprocessor import GroupedDataPreprocessor

class StockNormalizeFeaturesPreprocessor(GroupedDataPreprocessor):
    def __init__(self, normalize_columns):
        super().__init__()
        self.normalize_columns = normalize_columns
        self.scalers = {}

    def get_scaler(self, group_key):
        if not group_key in self.scalers:
            self.scalers[group_key] = preprocessing.StandardScaler()
        return self.scalers[group_key]
    
    def fit(self, df_grouped_map):
        for group_key, df_grouped in df_grouped_map.items():
            scaler = self.get_scaler(group_key)
            scaler.fit(df_grouped[self.normalize_columns])

    def apply(self, df_grouped_map):
        final_df_grouped_map = {}
        for group_key, df_grouped in df_grouped_map.items():
            scaler = self.get_scaler(group_key)
            row_index = df_grouped.index
            features = df_grouped[self.normalize_columns]
            transformed = scaler.transform(features)
            df_grouped.loc[row_index, self.normalize_columns] = transformed
            final_df_grouped_map[group_key] = df_grouped
        return final_df_grouped_map
