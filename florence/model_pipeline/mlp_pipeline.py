import traceback

from sklearn.neural_network import MLPRegressor
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras import applications
from keras import Input
from keras import Model
from keras import Sequential
from keras import losses

class MLPModelPipeline(ModelPipeline):

    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.count = 0

    def init_model(self, param: dict = None):

        # self.model = models.Sequential()
        # self.model.add(layers.Dense(128, activation='sigmoid', input_shape=(13)))
        # self.model.add(layers.Dense(256, activation='sigmoid'))
        # self.model.add(layers.Dense(128, activation='sigmoid'))
        # self.model.add(layers.Dense(1, activation='softmax'))
        self.model = Sequential(
        [
            layers.Input(shape=(12,)),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(256, activation='sigmoid'),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(1),
            # layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            # layers.GlobalAveragePooling2D(),
            # layers.Dropout(0.5),
            # layers.Dense(num_classes, activation="softmax"),
            ]   
        )
        print(self.model.summary())
        self.model.compile(
            optimizer=optimizers.RMSprop(learning_rate=0.1),
            loss=losses.MeanAbsoluteError(),
            metrics=['mae']
        )
        # self.model = model


    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        # print(train_X[:10])
        # print(train_Y[:10])
        # print(eval_X[:10])
        # print(eval_Y[:10])
        try:
            history = self.model.fit(
                train_X,
                train_Y,
                validation_data=(eval_X, eval_Y),
                epochs=10,
            )
        except Exception as ex:
            print("fit exception")
            traceback.print_exc()
            raise ex
        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        # self.history_list.add(history)
        # plt.plot(epochs, acc, 'bo', label='Training acc')
        # plt.plot(epochs, val_acc, 'b', label='Validation acc')
        # plt.title('Training and validation accuracy')
        # plt.legend()        
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # plt.show()

        # Show the plot
        plt.savefig(f'img/mlp_{self.model_id}_loss_{self.count}.jpg')
        self.count+=1
        # shap_img = plt.imread(f"img/shap_{save_name}.jpg")
        # plt.imshow(shap_img)



    def get_name(self):
        return "mlp"


    # def get_name_with_params(self, params):
    #     selected_params_for_model_id = ['learning_rate', 'max_depth', 'n_estimators']
    #     return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_id])

    # def get_hyper_params(self, trial):
    #     return {
    #         'hidden_layer_sizes' : [128, 256, 128],
    #         # 'activation' : 'relu',
    #         # 'solver' : 'adam',
    #         # 'alpha' : 0.0,
    #         'batch_size' : 10,
    #         # 'random_state' : 0,
    #         # 'tol' : 0.0001,
    #         # 'nesterovs_momentum' : False,
    #         # 'learning_rate' : 'constant',
    #         'learning_rate_init' : trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
    #         'max_iter' : 200,
    #         # 'shuffle' : True,
    #         'n_iter_no_change' : 10,
    #         'early_stopping': True,
    #         # 'verbose' : False 
    #     }

class MLPModelPipelineFactory(ModelPipelineFactory):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id

    def create_model_pipeline(self) -> ModelPipeline:
        return MLPModelPipeline(self.model_id)