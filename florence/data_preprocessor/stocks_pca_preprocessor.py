import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from itertools import combinations
from data_preprocessor.data_preprocessor import DataPreprocessor
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
import time

class StocksPcaPreProcessor(DataPreprocessor):
    def _init_(self):
        super()._init_()

    def apply(self, df):
        stock_clusters = pd.read_csv('./stocks_pca/stock_clusters.csv')
        df = df.merge(stock_clusters, how='left', on='stock_id')
        
        return df