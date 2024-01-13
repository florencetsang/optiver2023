import numpy as np
import pandas as pd
from itertools import combinations
from data_preprocessor.data_preprocessor import DataPreprocessor

class EnrichDFDataPreprocessor(DataPreprocessor):
    def calculate_pressure(self, df):
        return np.where(
            df['imbalance_buy_sell_flag']==1,
            df['imbalance_size']/df['ask_size'],
            df['imbalance_size']/df['bid_size']
        )
    
    def apply(self, df):
        prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
        sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

        df_ = df.copy()

        # Imbalance features
        df_['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
        df_['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')  

        for c in combinations(prices, 2):
            df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

        for a, b, c in combinations( ['reference_price', 'ask_price', 'bid_price', 'wap'], 3):
            maxi = df_[[a,b,c]].max(axis=1)
            mini = df_[[a,b,c]].min(axis=1)
            mid = df_[[a,b,c]].sum(axis=1)-mini-maxi

            df_[f'{a}_{b}_{c}_imb2'] = np.where(mid.eq(mini), np.nan, (maxi - mid) / (mid - mini))

        # From Andy - Pressure & Inefficiency
        df_['pressure'] = self.calculate_pressure(df)
        df_['inefficiency'] = df.eval('imbalance_size/matched_size')

        return df_

class MovingAvgPreProcessor(DataPreprocessor):
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name
    
    def _get_mov_avg_col_name(self, window, min_periods):
        # col_name = self._get_col_name(window, min_periods)
        return f"{self.feature_name}_mov_avg_{window}_{min_periods}"  # e.g. wap_mov_avg_3_1

    def _calc_mov_avg2(self, orig_df, grouped_df, window, min_periods):
        mov_avg_col_name = self._get_mov_avg_col_name(window, min_periods)
        mov_avg_df = grouped_df[self.feature_name].rolling(window=window, min_periods=min_periods).mean()
        mov_avg_df = mov_avg_df.sort_index()
        orig_df[mov_avg_col_name] = mov_avg_df[self.feature_name]
        return orig_df

    def apply(self, df):
        # as_index=False --> SQL-like output with group keys as columns without nesting structure
        # sort=False --> Sort group keys. Get better performance by turning this off.
        # consolidated_mov_avg_df = pd.DataFrame()
        grouped_df = df.groupby(['stock_id', 'date_id'], as_index=False, sort=False)
        df = self._calc_mov_avg2(df, grouped_df, 3, 1)
        df = self._calc_mov_avg2(df, grouped_df, 6, 3)
        df = self._calc_mov_avg2(df, grouped_df, 12, 6)
        df = self._calc_mov_avg2(df, grouped_df, 24, 12)
        return df

class RemoveIrrelevantFeaturesDataPreprocessor(DataPreprocessor):
    def __init__(self, non_features):
        super().__init__()
        self.non_features = non_features
    
    def apply(self, df):
        useful_features = [c for c in df.columns if c not in self.non_features]
        processed_df = df[useful_features]
        return processed_df

class DropTargetNADataPreprocessor(DataPreprocessor):
    def __init__(self, target_col_name='target'):
        super().__init__()
        self.target_col_name = target_col_name
    
    def apply(self, df):
        processed_df = df.dropna(subset=[self.target_col_name])
        return processed_df
