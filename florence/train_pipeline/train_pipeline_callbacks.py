from utils.scoring_utils import ScoringUtils

class TrainPipelineCallback:
    def on_callback(self, models, model_res, train_dfs, eval_dfs, num_train_eval_sets):
        return None

class MAECallback(TrainPipelineCallback):
    def on_callback(self, models, model_res, train_dfs, eval_dfs, num_train_eval_sets):
        avg_mae = ScoringUtils.calculate_mae(models, eval_dfs)
        return avg_mae
