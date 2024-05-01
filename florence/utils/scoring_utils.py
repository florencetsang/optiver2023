import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from utils.ml_utils import MLUtils

class ScoringUtils:
    @staticmethod
    def map_to_score(fold_model, fold_eval_df):
        fold_eval_X, fold_eval_Y = MLUtils.create_XY(fold_eval_df)
        fold_pred = fold_model.predict(fold_eval_X)
        mae = mean_absolute_error(fold_eval_Y, fold_pred)
        return mae
    
    @staticmethod
    def calculate_mae(models, eval_dfs):
        maes = list(map(ScoringUtils.map_to_score, models, eval_dfs))
        avg_mae = np.average(maes)
        return avg_mae
