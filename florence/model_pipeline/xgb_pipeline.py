import xgboost as xgb
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class XGBModelPipeline(ModelPipeline):
  
    def __init__(self):
        super().__init__()        
        
    def init_model(self, param: dict = None):
        if param:
            self.model = xgb.XGBRegressor(**param)
        else:
            self.model = xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500,
                                          early_stopping_rounds=100, random_state=42)


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


    def get_name_with_params(self, params):
        selected_params_for_model_name = ['n_estimators', 'max_depth', 'subsample']
        return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_name])

    def get_hyper_params(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000, step=100),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "verbosity": 0,
            "objective": "reg:absoluteerror",
            # use exact for small dataset.
            "early_stopping_rounds": 100,
            # defines booster, gblinear for linear functions.
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "random_state": 42,
            "device": "cuda"
        }

class XGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return XGBModelPipeline()
