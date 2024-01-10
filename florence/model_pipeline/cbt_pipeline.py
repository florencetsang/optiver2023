import catboost as cbt
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class CatBoostModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
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
        
    def get_name(self):
        return "cbt"

class CatBoostModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return CatBoostModelPipeline()
