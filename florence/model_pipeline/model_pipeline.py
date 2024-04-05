from utils.ml_utils import MLUtils

class ModelPipeline:
    model = None

    def init_model(self):
        pass
    
    def train(self, df_train, df_eval):
        train_X, train_Y, eval_X, eval_Y = self.create_XY(df_train, df_eval)
        res = self._train(train_X, train_Y, eval_X, eval_Y)
        return res
    
    def create_XY(self, df_train, df_eval):
        train_X, train_Y = MLUtils.create_XY(df_train)
        eval_X, eval_Y = MLUtils.create_XY(df_eval)
        return train_X, train_Y, eval_X, eval_Y
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        return None
    
    def _get_eval_set(self, eval_X, eval_Y):
        if eval_X is None or eval_Y is None:
            return None
        return [(eval_X, eval_Y)]

    def get_model(self):
        return self.model

    def get_name(self):
        return "AbstractModelPipeline"

    def get_hyper_params(self, trial):
        return None

    def get_name_with_params(self, params):
        return None

class ModelPipelineFactory:
    def create_model_pipeline(self) -> ModelPipeline:
        return None
