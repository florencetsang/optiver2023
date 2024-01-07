import lightgbm as lgb
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class LGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
        self.model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=50)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        self.model.fit(train_X, train_Y, 
                eval_set=[(eval_X, eval_Y)], 
                callbacks=[lgb.early_stopping(100)]
            )
    
    def get_name(self):
        return "lgb"

class LGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return LGBModelPipeline()
