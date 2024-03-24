import catboost as cbt
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class CatBoostModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self, param: dict = None):
        if param:
            self.model = cbt.CatBoostRegressor(**param)
        else:
            self.model = cbt.CatBoostRegressor(objective='MAE', iterations=50)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        eval_set = self._get_eval_set(eval_X, eval_Y)
        self.model.fit(
            train_X,
            train_Y,
            eval_set=eval_set,
            early_stopping_rounds = 100
        )
        return None

    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        self.model.fit(
            train_X,
            train_Y,
            eval_set=[(eval_X, eval_Y)],
            early_stopping_rounds=100)
        
    def get_name(self):
        return "cbt"

    def get_hyper_params(self, trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "iterations": 100,
        }


class CatBoostModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return CatBoostModelPipeline()
