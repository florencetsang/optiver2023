from ml_utils import MLUtils

class ModelPipeline:
    model = None

    def init_model(self):
        pass
    
    def train(self, df_train, df_eval):
        train_X, train_Y = MLUtils.create_XY(df_train)
        eval_X, eval_Y = MLUtils.create_XY(df_eval)
        self._train(train_X, train_Y, eval_X, eval_Y)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        pass

    def get_model(self):
        return self.model

    def get_name(self):
        return "AbstractModelPipeline"

class ModelPipelineFactory:
    def create_model_pipeline(self) -> ModelPipeline:
        return None
