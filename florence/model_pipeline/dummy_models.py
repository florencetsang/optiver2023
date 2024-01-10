import numpy as np
from sklearn.base import BaseEstimator

class BaselineEstimator(BaseEstimator):
    def predict(self, X):
        return np.zeros(len(X))
    
# class SimpleEstimator(BaseEstimator):

#     simple_mapping = {
#         1: 0.1,
#         0: 0,
#         -1: -0.1
#     }

#     def predict(self, X):
#         return X['imbalance_buy_sell_flag'].map(self.simple_mapping)