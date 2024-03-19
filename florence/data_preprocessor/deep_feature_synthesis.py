import numpy as np
import pandas as pd
import featuretools as ft
from data_preprocessor.data_preprocessor import DataPreprocessor

class DfsPreProcessor(DataPreprocessor):
    def apply(self, df):

        df_ = df.copy()

        es = ft.EntitySet(id = 'train_df')
        es = es.entity_from_dataframe(entity_id = 'df', dataframe = df_, index = 'row_id')

        default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
        default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

        feature_names = ft.dfs(entityset = es, target_entity = 'df',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       max_depth = 2, features_only=True)
        
        print(feature_names)        

        return df_