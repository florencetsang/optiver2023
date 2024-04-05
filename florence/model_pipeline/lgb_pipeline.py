import lightgbm as lgb
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory
class LGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()

    def init_model(self, param: dict = None):
        if param:
            self.model = lgb.LGBMRegressor(**param)
        else:
            self.model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=50, random_state=42)

    def _train(self, train_X, train_Y, eval_X, eval_Y):
        eval_res = {}
        eval_set = self._get_eval_set(eval_X, eval_Y)
        self.model.fit(
            train_X,
            train_Y, 
            eval_set=eval_set,
            eval_metric='l1',
            callbacks=[
                lgb.early_stopping(100),
                lgb.record_evaluation(eval_res)
            ]
        )
        return eval_res

    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        self.model.fit(train_X, train_Y, eval_set=[(eval_X, eval_Y)], eval_metric='l1',
                                      callbacks=[
                                          lgb.early_stopping(stopping_rounds=100),
                                          lgb.record_evaluation(eval_res)
                                      ])
    
    def get_name(self):
        return "lgb"


    def get_name_with_params(self, params):
        selected_params_for_model_name = ['learning_rate', 'max_depth', 'n_estimators']
        return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_name])

    def get_hyper_params(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'n_estimators': trial.suggest_int('n_estimators', 100, 5000, step=100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 31, 512),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            # 'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
            'objective': 'regression_l1',
            'random_state': 42,
            'force_col_wise': True,
            "verbosity": -1,
        }

class LGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return LGBModelPipeline()
