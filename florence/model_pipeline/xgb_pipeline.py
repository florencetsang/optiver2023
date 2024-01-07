import xgboost as xgb
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class XGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
        self.model = xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500, early_stopping_rounds = 100)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        eval_set = self._get_eval_set(eval_X, eval_Y)
        self.model.fit(
            train_X,
            train_Y,
            eval_set=eval_set,
        )
        return None
        
    def get_name(self):
        return "xgb"

class XGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return XGBModelPipeline()
