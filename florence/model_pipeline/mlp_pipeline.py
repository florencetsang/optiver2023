from sklearn.neural_network import MLPRegressor
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

class MLPModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()

    def init_model(self, param: dict = None):
        self.model = MLPRegressor(**param)

    def _train(self, train_X, train_Y, eval_X, eval_Y):
        eval_res = {}
        eval_set = self._get_eval_set(eval_X, eval_Y)
        self.model.fit(
            train_X,
            train_Y, 
            eval_set=eval_set,
            eval_metric='l1',
        )
        return eval_res

    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        self.model.fit(train_X, train_Y)
    
    def get_name(self):
        return "mlp"


    def get_name_with_params(self, params):
        selected_params_for_model_name = ['learning_rate', 'max_depth', 'n_estimators']
        return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_name])

    def get_hyper_params(self, trial):
        return {
            'hidden_layer_sizes' : [256, 1024, 512, 256, 128],
            'activation' : 'relu',
            'solver' : 'adam',
            # 'alpha' : 0.0,
            # 'batch_size' : 10,
            'random_state' : 0,
            # 'tol' : 0.0001,
            # 'nesterovs_momentum' : False,
            # 'learning_rate' : 'constant',
            'learning_rate_init' : trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
            'max_iter' : trial.suggest_int('max_iter', 200, 1000, log=True),
            'shuffle' : True,
            'n_iter_no_change' : trial.suggest_int('n_iter_no_change', 10, 50, log=True),
            'early_stopping': True,
            # 'verbose' : False 
        }

class MLPModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return MLPModelPipeline()
