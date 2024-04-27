import numpy as np
import pandas as pd
import featuretools as ft
from data_preprocessor.data_preprocessor import DataPreprocessor
from woodwork.logical_types import Categorical

class StockDateIdPreprocessor(DataPreprocessor):
    def apply(self, df):
        df["stock_date_id"] = df["date_id"] * 1000 + df["stock_id"]
        return df

class FeatureToolsDFSPreprocessor(DataPreprocessor):
    def __init__(self, group_by_stock=True, group_by_date=True, group_by_stock_date=True):
        self.group_by_stock = group_by_stock
        self.group_by_date = group_by_date
        self.group_by_stock_date = group_by_stock_date
        assert group_by_stock or group_by_date or group_by_stock_date

        self.default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean"]
        self.default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

    def apply(self, df):
        df_ = df.copy(deep=True)

        es = ft.EntitySet(id = 'closing_movements_data')
        es = es.add_dataframe(
            dataframe_name="closing_movements",
            dataframe=df,
            index="row_id",
            time_index="time_id",
            logical_types={
                "imbalance_buy_sell_flag": Categorical,
                "stock_id": Categorical,
                "date_id": Categorical,
                "stock_date_id": Categorical,
            },
        )

        print("dfs - normalize_dataframe - start")
        if self.group_by_stock:
            print("group_by_stock - normalize_dataframe")
            es.normalize_dataframe("closing_movements", "stocks", "stock_id")
        if self.group_by_date:
            print("group_by_date - normalize_dataframe")
            es.normalize_dataframe("closing_movements", "date_ids", "date_id")
        if self.group_by_stock_date:
            print("group_by_stock_date - normalize_dataframe")
            es.normalize_dataframe("closing_movements", "stock_date_ids", "stock_date_id")
        print("dfs - normalize_dataframe - end")

        print(f"dfs - es: {es}")
        print(f"dfs - es closing_movements schema: {es['closing_movements'].ww.schema}")

        print("dfs - generate features - start")
        if self.group_by_stock:
            feature_matrix_stock, feature_defs_stock = ft.dfs(
                entityset=es,
                target_dataframe_name="stocks",
                # trans_primitives = default_trans_primitives,
                agg_primitives=self.default_agg_primitives,
                max_depth=2,
                verbose=True,
            )
            self._print_feature_matrix(feature_matrix_stock, "feature_matrix_stock", "generate all features")
            feature_matrix_stock, feature_defs_stock = ft.selection.remove_single_value_features(feature_matrix_stock, features=feature_defs_stock)
            self._print_feature_matrix(feature_matrix_stock, "feature_matrix_stock", "remove_single_value_features")
            feature_matrix_stock, feature_defs_stock = ft.selection.remove_highly_correlated_features(feature_matrix_stock, features=feature_defs_stock)
            self._print_feature_matrix(feature_matrix_stock, "feature_matrix_stock", "remove_highly_correlated_features")

        if self.group_by_date:
            feature_matrix_date, feature_defs_date = ft.dfs(
                entityset=es,
                target_dataframe_name="date_ids",
                # trans_primitives = default_trans_primitives,
                agg_primitives=self.default_agg_primitives,
                max_depth=2,
                verbose=True,
            )
            self._print_feature_matrix(feature_matrix_date, "feature_matrix_date", "generate all features")
            feature_matrix_date, feature_defs_date = ft.selection.remove_single_value_features(feature_matrix_date, features=feature_defs_date)
            self._print_feature_matrix(feature_matrix_date, "feature_matrix_date", "remove_single_value_features")
            feature_matrix_date, feature_defs_date = ft.selection.remove_highly_correlated_features(feature_matrix_date, features=feature_defs_date)
            self._print_feature_matrix(feature_matrix_date, "feature_matrix_date", "remove_highly_correlated_features")

        if self.group_by_stock_date:
            feature_matrix_stock_date, feature_defs_stock_date = ft.dfs(
                entityset=es,
                target_dataframe_name="stock_date_ids",
                # trans_primitives = default_trans_primitives,
                agg_primitives=self.default_agg_primitives,
                max_depth=2,
                verbose=True,
            )
            self._print_feature_matrix(feature_matrix_stock_date, "feature_matrix_stock_date", "generate all features")
            feature_matrix_stock_date, feature_defs_stock_date = ft.selection.remove_single_value_features(feature_matrix_stock_date, features=feature_defs_stock_date)
            self._print_feature_matrix(feature_matrix_stock_date, "feature_matrix_stock_date", "remove_single_value_features")
            feature_matrix_stock_date, feature_defs_stock_date = ft.selection.remove_highly_correlated_features(feature_matrix_stock_date, features=feature_defs_stock_date)
            self._print_feature_matrix(feature_matrix_stock_date, "feature_matrix_stock_date", "remove_highly_correlated_features")
        print("dfs - generate features - start")

        print(f"before dfs - shape: {df_}, memory usage: {df_.memory_usage(index=True).sum() / 1024 / 1024}")
        if self.group_by_stock:
            df_ = df_.merge(feature_matrix_stock, left_on="stock_id", right_on="stock_id", how="left", suffixes=("dfs_stock_left", "dfs_stock_right"))
            print(f"dfs merge to df - group_by_stock - {df_}")
        if self.group_by_date:
            df_ = df_.merge(feature_matrix_date, left_on="date_id", right_on="date_id", how="left", suffixes=("dfs_date_left", "dfs_date_right"))
            print(f"dfs merge to df - group_by_date - {df_}")
        if self.group_by_stock_date:
            df_ = df_.merge(feature_matrix_stock_date, left_on="stock_date_id", right_on="stock_date_id", how="left", suffixes=("dfs_stock_date_left", "dfs_stock_date_right"))
            print(f"dfs merge to df - group_by_stock_date - {df_}")
        print(f"after dfs - shape: {df_}, memory usage: {df_.memory_usage(index=True).sum() / 1024 / 1024}")

        return df_

    def _print_feature_matrix(self, feature_matrix, name, step):
        print(f"{name} - {step} - shape: {feature_matrix.shape}, memory usage: {feature_matrix.memory_usage(index=True).sum() / 1024 / 1024}")

class DfsPreProcessor():
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