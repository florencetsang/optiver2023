import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import TransformerMixin, BaseEstimator
from data_preprocessor.data_preprocessor import DataPreprocessor
from utils.dataframe_utils import get_df_summary_str

class StockIdFeaturesPreProcessor(DataPreprocessor):
    def __init__(self, target_col_name='target'):
        super().__init__()
        self.target_col_name = target_col_name

    def apply(self, df):

        df_ = df.copy()

        global_stock_id_feats = {
            "median_size": df_.groupby("stock_id")["bid_size"].median() + df_.groupby("stock_id")["ask_size"].median(),
            "std_size": df_.groupby("stock_id")["bid_size"].std() + df_.groupby("stock_id")["ask_size"].std(),
            "ptp_size": df_.groupby("stock_id")["bid_size"].max() - df_.groupby("stock_id")["bid_size"].min(),
            "median_price": df_.groupby("stock_id")["bid_price"].median() + df_.groupby("stock_id")["ask_price"].median(),
            "std_price": df_.groupby("stock_id")["bid_price"].std() + df_.groupby("stock_id")["ask_price"].std(),
            "ptp_price": df_.groupby("stock_id")["bid_price"].max() - df_.groupby("stock_id")["ask_price"].min(),
        }        

        for key, value in global_stock_id_feats.items():
            df_[f"global_{key}"] = df_["stock_id"].map(value.to_dict())

        return df_


class StockIdFeaturesDataTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, target_col_name='target'):
        super().__init__()
        self.target_col_name = target_col_name
        self.global_stock_id_feats = None

    def fit(self, df, y=None):
        grouped = df.groupby("stock_id")
        print(f"StockIdFeaturesDataTransformer - fit start")
        self.global_stock_id_feats = {}
        self.global_stock_id_feats["median_size"] = grouped["bid_size"].median() + grouped["ask_size"].median()
        self.global_stock_id_feats["std_size"] = grouped["bid_size"].std() + grouped["ask_size"].std()
        self.global_stock_id_feats["ptp_size"] = grouped["bid_size"].max() - grouped["bid_size"].min()
        self.global_stock_id_feats["median_price"] = grouped["bid_price"].median() + grouped["ask_price"].median()
        self.global_stock_id_feats["std_price"] = grouped["bid_price"].std() + grouped["ask_price"].std()
        self.global_stock_id_feats["ptp_price"] = grouped["bid_price"].max() - grouped["ask_price"].min()
        print(f"StockIdFeaturesDataTransformer - fit end")
        return self
    
    def transform(self, df):
        print(f"StockIdFeaturesDataTransformer - transform start - df: {get_df_summary_str(df)}")
        df_ = df.copy()
        for key, value in self.global_stock_id_feats.items():
            df_[f"global_{key}"] = df_["stock_id"].map(value.to_dict())
        print(f"StockIdFeaturesDataTransformer - transform end - df: {get_df_summary_str(df_)}")
        return df_
