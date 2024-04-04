import numpy as np
import pandas as pd
import featuretools as ft
from data_preprocessor.data_preprocessor import DataPreprocessor
from woodwork.logical_types import Categorical
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)

class DfsPreProcessor(DataPreprocessor):
    def apply(self, df):

        df_ = df.copy()
        es = ft.EntitySet(id = 'train_df')

        # add original dataframe
        es = es.add_dataframe(
            dataframe_name="closing_movements",
            dataframe=df_,
            index="row_id",
            time_index="time_id",
            logical_types={
                "imbalance_buy_sell_flag": Categorical,
            },
        )

        # add stocks df
        stocks_df = pd.DataFrame()
        stocks_df["stock_id"] = pd.Series(pd.unique(df_["stock_id"]))
        stocks_df["dummy"] = pd.Series(pd.unique(df_["stock_id"]))
        es = es.add_dataframe(
            dataframe_name="stocks", dataframe=stocks_df, index="stock_id"
        )

        es = es.add_relationship("stocks", "stock_id", "closing_movements", "stock_id")

        default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean"]

        feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                    target_dataframe_name="stocks",
                                    # trans_primitives = default_trans_primitives,
                                    agg_primitives=default_agg_primitives, 
                                    max_depth = 2)
        
        print(f'Features: {feature_matrix.columns}')
        
        # feature selection        
        # feature_matrix, feature_defs = remove_highly_null_features(/feature_matrix)
        # print(f'Features after remove null: {feature_matrix.columns}')
        feature_matrix, feature_defs = remove_single_value_features(feature_matrix, features=feature_defs)
        print(f'Features after remove sinlge value: {feature_matrix.columns}')
        feature_matrix, feature_defs = remove_highly_correlated_features(feature_matrix, features=feature_defs)
        print(f'Features after removing highly correlated features: {feature_matrix.columns}')

        feature_matrix.drop(['dummy'], axis = 1)     

        return df_.merge(feature_matrix, left_on = "stock_id", right_on = "stock_id", how = "left")