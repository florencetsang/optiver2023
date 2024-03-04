import numpy as np
import pandas as pd
from itertools import combinations
from data_preprocessor.data_preprocessor import DataPreprocessor
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
import time
import matplotlib.pyplot as plt
class BasicFeaturesPreprocessor(DataPreprocessor):
    def calculate_pressure(self, df):
        return np.where(
            df['imbalance_buy_sell_flag']==1,
            df['imbalance_size']/df['ask_size'],
            df['imbalance_size']/df['bid_size']
        )
    
    def apply(self, df):
        df_ = df.copy()

        # Imbalance features
        df_['bid_ask_rr'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
        df_['shortage_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')  

        # From Andy - Pressure & Inefficiency
        df_['pressure'] = self.calculate_pressure(df)
        df_['shortage_s1'] = df.eval('imbalance_size/matched_size')

        return df_
    
class DupletsTripletsPreprocessor(DataPreprocessor):

    def apply(self, df):

        prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
        df_ = df.copy()

        for c in combinations(prices, 2):
            df_[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

        for a, b, c in combinations( ['reference_price', 'ask_price', 'bid_price', 'wap'], 3):
            maxi = df_[[a,b,c]].max(axis=1)
            mini = df_[[a,b,c]].min(axis=1)
            mid = df_[[a,b,c]].sum(axis=1)-mini-maxi

            df_[f'{a}_{b}_{c}_imb2'] = np.where(mid.eq(mini), np.nan, (maxi - mid) / (mid - mini))

        return df_

class MovingAvgPreProcessor(DataPreprocessor):
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name
    
    def _get_mov_avg_col_name(self, window, min_periods):
        # col_name = self._get_col_name(window, min_periods)
        return f"{self.feature_name}_mov_avg_{window}_{min_periods}"  # e.g. wap_mov_avg_3_1

    def _calc_mov_avg(self, orig_df, grouped_df, window, min_periods):
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
        df = self._calc_mov_avg(df, grouped_df, 3, 1)
        df = self._calc_mov_avg(df, grouped_df, 6, 3)
        df = self._calc_mov_avg(df, grouped_df, 12, 6)
        df = self._calc_mov_avg(df, grouped_df, 24, 12)
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

class DTWKMeansPreprocessor(DataPreprocessor):

    def __init__(self, n_clusters=5, target_col_name='wap'):
        self.n_clusters = n_clusters
        self.target_col_name = target_col_name
        self.model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    def fit(self, time_series_data, max_clusters=10):
        inertias = []
        for n_clusters in range(1, max_clusters + 1):
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
            model.fit(time_series_data)
            inertias.append(model.inertia_)
        return inertias
    def data_manipulation(self, df):
        df.set_index(['date_id', 'time_id', 'stock_id'], inplace=True)
        df_unstacked = df.unstack(level='stock_id')
        df_unstacked.columns = ['_'.join(map(str, col)).strip() for col in df_unstacked.columns.values]
        pivoted_df = df_unstacked.fillna(0)
        pivoted_df .reset_index(inplace=True)
        relevant_columns = [col for col in pivoted_df.columns if col.startswith(self.target_col_name)]
        relevant_columns =relevant_columns
        pivoted_df = pivoted_df[relevant_columns]
        time_series_data = pivoted_df.to_numpy()
        return time_series_data

    def apply(self, df):
        print("DTWKMeansPreprocessor_start")
        df.set_index(['date_id', 'time_id', 'stock_id'], inplace=True)
        df_unstacked = df.unstack(level='stock_id')
        df_unstacked.columns = ['_'.join(map(str, col)).strip() for col in df_unstacked.columns.values]
        pivoted_df = df_unstacked.fillna(0)
        pivoted_df .reset_index(inplace=True)
        relevant_columns = [col for col in pivoted_df.columns if col.startswith(self.target_col_name)]
        relevant_columns =relevant_columns
        pivoted_df = pivoted_df[relevant_columns]
        time_series_data = pivoted_df.to_numpy()
        labels = self.model.fit_predict(time_series_data)
        cluster_df = pd.DataFrame(labels, index=df_unstacked.index, columns=['cluster'])
        df.reset_index(inplace=True)
        processed_df = pd.merge(df, cluster_df, on=['date_id', 'time_id'], how='left')
        processed_df = processed_df.dropna(subset=['cluster'])
        processed_df['cluster'] = processed_df['cluster'].astype(int)

        print("DTWKMeansPreprocessor_end")
        return processed_df