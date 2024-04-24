from os import path
import traceback

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory
import matplotlib.pyplot as plt

from keras import layers
from keras import optimizers
from keras import Sequential
from keras import losses

class MLPModelPipeline(ModelPipeline):

    layer_choices = [
        [64,64,64],
        [128,64,32],
        [64,32,16],
        [64,128,64],
    ]

    def __init__(self, model_id, plot_path, num_features=12):
        super().__init__()
        self.model_id = model_id
        self.plot_path = plot_path
        self.num_features = num_features

    def init_model(self, param: dict = None, fold=9999):
        # init a new model + reset stateful params
        self.reset()
        
        self.param = param

        # self.model = models.Sequential()
        # self.model.add(layers.Dense(128, activation='sigmoid', input_shape=(13)))
        # self.model.add(layers.Dense(256, activation='sigmoid'))
        # self.model.add(layers.Dense(128, activation='sigmoid'))
        # self.model.add(layers.Dense(1, activation='softmax'))

        self.model = Sequential()
        self.model.add(layers.Input(shape=(self.num_features,)))
        layer_configs = self.layer_choices[self.param["layers"]]
        for layer_config in layer_configs:
            self.model.add(layers.Dense(layer_config, activation='relu'))
        self.model.add(layers.Dense(1))


        # self.model = Sequential(
        # [
        #     # relu
        #     layers.Input(shape=(self.num_features,)),
        #     layers.Dense(self.param["layers"][0], activation='relu'),
        #     layers.Dense(self.param["layers"][1], activation='relu'),
        #     layers.Dense(self.param["layers"][2], activation='relu'),
        #     layers.Dense(1),
        #     # leaky relu
        #     # layers.Input(shape=(self.num_features,)),
        #     # layers.Dense(128),
        #     # layers.LeakyReLU(),
        #     # layers.Dense(64),
        #     # layers.LeakyReLU(),
        #     # layers.Dense(32),
        #     # layers.LeakyReLU(),
        #     # layers.Dense(1),
        #     ]   
        # )
        print(self.model.summary())
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.param["learning_rate"]),
            loss=losses.MeanAbsoluteError(),
            metrics=['mae']
        )
        self.fold = fold

    def reset(self):
        self.history = None

    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        print(f"train - train_X: {train_X.shape}, train_Y: {train_Y.shape}, eval_X: {eval_X.shape}, eval_Y: {eval_Y.shape}")
        try:
            self.history = self.model.fit(
                train_X,
                train_Y,
                validation_data=(eval_X, eval_Y),
                epochs=self.param["epochs"],
                batch_size=self.param["batch_size"],
            )
        except Exception as ex:
            print("fit exception")
            traceback.print_exc()
            raise ex
    
    def eval_once(self, x, y):
        pred = self.model.predict(x, batch_size=256)
        mae = mean_absolute_error(y, pred)
        return mae

    def get_name(self):
        return "mlp"

    # def get_name_with_params(self, params):
    #     selected_params_for_model_id = ['learning_rate', 'max_depth', 'n_estimators']
    #     return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_id])

    def get_static_params(self):
        return {
            'epochs': 10,
            'batch_size': 256,
        }

    def post_train(self):
        trial_id = self.param["trial_id"] if "trial_id" in self.param else "trialDefault"
        loss_plot_filename = path.join(self.plot_path, f'mlp_{self.model_id}_optuna{trial_id}_loss_{self.fold}.jpg')
        print(f"saving loss plot to {loss_plot_filename}")
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # plt.show()
        plt.savefig(loss_plot_filename)


    def get_hyper_params(self, trial):
        hyper_params_dict = self.get_static_params()
        hyper_params_dict.update(
            {
                "trial_id": trial.number,
                'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.0004, log=True),
                'layers': trial.suggest_categorical('layers', list(range(len(self.layer_choices)))),
            }
        )
        return hyper_params_dict


class MLPModelPipelineFactory(ModelPipelineFactory):
    def __init__(self, model_id, plot_path, num_features):
        super().__init__()
        self.model_id = model_id
        self.plot_path = plot_path
        self.num_features = num_features

    def create_model_pipeline(self) -> ModelPipeline:
        return MLPModelPipeline(self.model_id, self.plot_path, self.num_features)
