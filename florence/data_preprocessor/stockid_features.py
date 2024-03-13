import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from data_preprocessor.data_preprocessor import DataPreprocessor

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