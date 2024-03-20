import numpy as np
import pandas as pd
from itertools import combinations
from data_preprocessor.data_preprocessor import GroupedDataPreprocessor

class StockNormalizeFeaturesPreprocessor(GroupedDataPreprocessor):
    def __init__(self, normalize_columns):
        super().__init__()
        self.normalize_columns = normalize_columns
    
    def apply(self, df, df_grouped):
        df[self.normalize_columns] = df_grouped[self.normalize_columns].transform(lambda x: (x - x.mean()) / x.std())
        return df, df_grouped
