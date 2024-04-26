import numpy as np
import pandas as pd
from itertools import combinations
from data_preprocessor.data_preprocessor import DataPreprocessor
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
import time

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
    def __init__(self, enable_duplets=True, enable_triplets=True):
        super().__init__()
        self.enable_duplets = enable_duplets
        self.enable_triplets = enable_triplets

    def apply(self, df):

        prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
        df_ = df.copy()

        if self.enable_duplets:
            for c in combinations(prices, 2):
                df_[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

        if self.enable_triplets:
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
        mov_avg_df = grouped_df[self.feature_name].rolling(window=window, min_periods=min_periods).mean().astype(np.float32)
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
    

#to implement EWMA

class EWMAPreProcessor(DataPreprocessor):
    def __init__(self, feature_name, span):
        super().__init__()
        self.feature_name = feature_name
        self.span = span
    
    def _get_ewma_col_name(self):
        return f"{self.feature_name}_ewma_{self.span}"  # e.g., wap_ewma_5
    
    def _calc_ewma(self, df):
        ewma_col_name = self._get_ewma_col_name()
        grouped_df = df.groupby(['stock_id', 'date_id'], as_index=False, sort=False)[self.feature_name]
        ewma_df = grouped_df.ewm(span=self.span, min_periods=0).mean()
        df[ewma_col_name] = ewma_df
        return df
    
    def apply(self, df):
        return self._calc_ewma(df)





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

class AddStockDateIdxDataPreprocessor(DataPreprocessor):
    def apply(self, df):
        # index_col_id: int
        # TODO: assumed # of stocks < 1000 now
        # https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-pandas-dataframe
        df["index_col_id"] = df["date_id"] * 1000 + df["stock_id"]
        return df

class FarNearPriceFillNaPreprocessor(DataPreprocessor):
    def apply(self, df):
        # TODO: other fillna logic?
        mask = df["far_price"].isna()
        df["far_price"] = df["far_price"].mask(mask, 1.0)
        mask = df["near_price"].isna()
        df["near_price"] = df["near_price"].mask(mask, 1.0)
        return df

class MovingAvgFillNaPreprocessor(DataPreprocessor):
    def __init__(self, feature_name, fill_na_value):
        super().__init__()
        self.feature_name = feature_name
        self.fill_na_value = fill_na_value
    
    def apply(self, df):
        # TODO: other fillna logic?
        columns = df.columns[df.columns.str.startswith(f"{self.feature_name}_mov_avg")]
        df[columns] = df[columns].fillna(self.fill_na_value)
        return df

class RemoveRecordsByStockDateIdPreprocessor(DataPreprocessor):
    def __init__(self, stock_date_keys):
        super().__init__()
        self.stock_date_keys = stock_date_keys

    def apply(self, df):
        keep_mask = np.ones(df.shape[0], dtype=bool)
        for stock_date_key in self.stock_date_keys:
            remove_mask = (df["stock_id"] == stock_date_key["stock_id"]) & (df["date_id"] == stock_date_key["date_id"])
            keep_mask = keep_mask & (~remove_mask)
        final_df = df.drop(df[~keep_mask].index)
        removed_records = df.shape[0] - final_df.shape[0]
        print(f"RemoveRecordsByStockDateIdPreprocessor - removing {removed_records} records")
        return final_df
class DTWKMeansPreprocessor(DataPreprocessor):

    def __init__(self, n_clusters=3, target_col_name='wap'):
        self.n_clusters = n_clusters
        self.target_col_name = target_col_name
        self.model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
        
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
