import lightgbm as lgb
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory
import optuna
from utils.scoring_utils import ScoringUtils
import numpy as np
import sklearn


class LGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()

    def init_model(self):
        self.model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=50, random_state=42)

    def _train(self, train_X, train_Y, eval_X, eval_Y):
        eval_res = {}
        eval_set = self._get_eval_set(eval_X, eval_Y)

        def objective(trial):
            # data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
            # train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
            dtrain = lgb.Dataset(train_X, label=train_Y)
            param = {
                "objective": "regression_l1",
                "metric": "mae",
                "n_estimators": 1000,
                "verbosity": -1,
                "bagging_freq": 1,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }

            gbm = lgb.train(param, dtrain)
            preds = gbm.predict(eval_X)
            pred_labels = np.rint(preds)
            mae = sklearn.metrics.mean_absolute_error(eval_Y, pred_labels)
            # mae = ScoringUtils.calculate_mae(gbm, eval_Y)
            return mae

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        # self.model.fit(
        #     train_X,
        #     train_Y,
        #     eval_set=eval_set,
        #     eval_metric='l1',
        #     callbacks=[
        #         lgb.early_stopping(100),
        #         lgb.record_evaluation(eval_res)
        #     ]
        # )
        return eval_res

    def get_name(self):
        return "lgb"


class LGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return LGBModelPipeline()
