import numpy as np 
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
  
        # Rolling based features
         # df_['wap_30'] = df.groupby(['stock_id', 'date_id'])['wap'].rolling(window=3,min_periods=3).mean()  
        df_['wap_30'] = df.groupby(['stock_id', 'date_id'])['wap'].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df_['wap_60'] = df.groupby(['stock_id', 'date_id'])['wap'].transform(lambda s: s.rolling(6, min_periods=3).mean())
        df_['wap_120'] = df.groupby(['stock_id', 'date_id'])['wap'].transform(lambda s: s.rolling(12, min_periods=6).mean())
        df_['wap_120'] = df.groupby(['stock_id', 'date_id'])['wap'].transform(lambda s: s.rolling(24, min_periods=12).mean())

        return df_

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
