from data_generator.data_generator import TrainEvalDataGenerator
from model_pipeline.model_pipeline import ModelPipelineFactory
from model_post_processor.model_post_processor import ModelPostProcessor

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import shap
import pandas as pd
import matplotlib.pyplot as plt

import joblib
import optuna
import time
import json

from os import path
import keras

class DefaultOptunaTrainPipeline():
    def __init__(
            self,
            model_pipeline_factory: ModelPipelineFactory,
            train_eval_data_generator: TrainEvalDataGenerator,
            model_post_processor: ModelPostProcessor,
            callbacks,
            num_trials=10,
    ):
        self.model_pipeline_factory = model_pipeline_factory
        self.model_pipeline = self.model_pipeline_factory.create_model_pipeline()
        self.train_eval_data_generator = train_eval_data_generator
        self.model_post_processor = model_post_processor
        self.callbacks = callbacks
        self.num_trials = num_trials

    def plot_feature_importance(self, lgbm_model, df_train, save_name):
        # feature importance
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        import textwrap  # pip install textwrap3
        # Set the feature names of the LightGBM model to be the columns of the encoded data frame (excluding the target column)
        lgbm_model.feature_names = df_train.drop('target', axis=1).columns

        # Get the feature names from the LightGBM model
        feature_names = lgbm_model.feature_names

        # Retrieve the feature importances from the LightGBM model
        importances = lgbm_model.feature_importances_

        # Normalize the feature importances to sum to 100
        normalized_importances = 100 * importances / np.sum(importances)

        # Create a DataFrame of the feature names and their normalized importances
        feat_df = pd.DataFrame({'feature': feature_names, 'importance': normalized_importances})

        # Sort the DataFrame by the importance values in descending order
        sorted_df = feat_df.sort_values('importance', ascending=False)

        # Create a bar plot of the top 10 most important features
        plt.rcdefaults()
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(y='importance', x='feature', data=sorted_df.head(10), palette='mako')
        plt.title('Feature Importance to predict price by LightGBM')
        plt.ylabel('Feature Importance (%)')

        # Add labels to the bars showing the percentage of importance for each feature
        ax.bar_label(ax.containers[0], fmt='%.2f')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
        max_width = 5
        ax.set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels())
        # Show the plot
        plt.savefig(f'img/feature_importance_{save_name}.jpg')
        feature_importance_img = plt.imread(f"img/feature_importance_{save_name}.jpg")
        plt.imshow(feature_importance_img)

    def plot_shap(self, lgbm_model, df_train, df_eval, save_name, shap_data_size):
        # shap
        # Create a SHAP explainer object for the LightGBM model
        explainer = shap.TreeExplainer(lgbm_model)

        # Sample 10% of the test data to use for faster computation of SHAP values
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = self.model_pipeline.create_XY(df_train, df_eval)
        sample = pd.DataFrame(X_val_fold).sample(frac=shap_data_size, random_state=42)

        # Calculate the SHAP values for the sampled test data using the explainer
        shap_values = explainer.shap_values(sample)

        # Create a SHAP summary plot of the feature importances based on the SHAP values
        feature_names = df_train.drop('target', axis=1).columns

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, sample, show=False, feature_names=feature_names)

        # Adjust the layout of the plot to avoid overlapping labels
        plt.tight_layout()

        # Show the plot
        plt.savefig(f'img/shap_{save_name}.jpg')
        shap_img = plt.imread(f"img/shap_{save_name}.jpg")
        plt.imshow(shap_img)

    def train(self, df_train):
        print(f"Generate data")
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)

        models = []
        model_res = []

        print(f"Start train and tune, num_train_eval_sets: {num_train_eval_sets}")
        tic = time.perf_counter() # Start Time

        def cross_validation_fcn(train_dfs, eval_dfs, param, early_stopping_flag=False):
            """
            Performs cross-validation on a given model using KFold and returns the average
            mean squared error (MSE) score across all folds.

            Parameters:
            - X_train: the training data to use for cross-validation
            - model: the machine learning model to use for cross-validation
            - early_stopping_flag: a boolean flag to indicate whether early stopping should be used

            Returns:
            - model: the trained machine learning model
            - mean_mse: the average MSE score across all folds
            """
            mse_list = []
            num_folds = len(train_dfs)
            for fold_index in range(num_folds):
                print(f"train fold {fold_index+1}/{num_folds} - start")
                
                # Split the data into training and validation sets
                X_train_fold, y_train_fold, X_val_fold, y_val_fold = self.model_pipeline.create_XY(train_dfs[fold_index], eval_dfs[fold_index])

                # create the LightGBM regressor with the optimized parameters
                self.model_pipeline.init_model(param=param)
                print(f"fold {fold_index+1}/{num_folds} - initialized params")
                # Train the model on the training set
                eval_res = {}
                # Use early stopping if enabled
                self.model_pipeline.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, eval_res)
                print(f"fold {fold_index+1}/{num_folds} - finished training")

                self.model_pipeline.post_train()
                print(f"fold {fold_index+1}/{num_folds} - finished post_train")

                # Make predictions on the validation set and calculate the MSE score
                y_pred = self.model_pipeline.model.predict(X_val_fold)
                mse = mean_absolute_error(y_val_fold, y_pred)
                mse_list.append(mse)
                print(f"fold {fold_index+1}/{num_folds} - mae: {mse}")
                
                print(f"train fold {fold_index+1}/{num_folds} - end")

            # Return the trained model and the average MSE score
            return self.model_pipeline.model, np.mean(mse_list)

        def objective(trial):
            print(f"optuna trial {trial.number+1}/{self.num_trials} - start")
            
            # set up the parameters to be optimized
            param = self.model_pipeline.get_hyper_params(trial)
            print(f"optuna trial {trial.number+1}/{self.num_trials} - params: {json.dumps(param, indent=2)}")

            # perform cross-validation using the optimized LightGBM regressor
            model, mean_score = cross_validation_fcn(train_dfs, eval_dfs, param, early_stopping_flag=True)
            print(f"optuna trial {trial.number+1}/{self.num_trials} - finished cross_validation_fcn")

            # retrieve the best iteration of the model and store it as a user attribute in the trial object
            if hasattr(model, 'best_iteration_'):
                best_iteration = model.best_iteration_
                trial.set_user_attr('best_iteration', best_iteration)
            else:
                print(f"model does not have best_iteration_ attribute")
            trial.set_user_attr('model', model)

            print(f"optuna trial {trial.number+1}/{self.num_trials} - end")

            return mean_score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.num_trials)

        print(f"Number of finished trials (total: {self.num_trials}): {len(study.trials)}")

        print("Best trial:")
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))

        toc = time.perf_counter() # End Time
        print(f"Finished train and tune. Took {(toc-tic):.2f}s.")

        # callback_results = []
        # for callback in self.callbacks:
        #     callback_res = callback.on_callback(models, model_res, train_dfs, eval_dfs, num_train_eval_sets)
        #     callback_results.append(callback_res)

        best_param = self.model_pipeline.get_static_params()

        best_param.update(best_trial.params)

        return best_param

    def train_with_param(self, df_train, params, model_name, model_type):
        print(f"Start training with params: {params}.")
        tic = time.perf_counter() # Start Time
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair
        train_dfs, eval_dfs = train_dfs[-1], eval_dfs[-1]

        # Fit the model to the training data
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = self.model_pipeline.create_XY(train_dfs, eval_dfs)
        self.model_pipeline.init_model(param=params)
        self.model_pipeline.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, {})

        best_model_name = self.model_pipeline.get_name_with_params(params)

        if model_type=="mlp":
            save_path = f"best_models/{self.model_pipeline.get_name()}_{best_model_name}_{model_name}.keras"
            self.model_pipeline.model.save(save_path)
        else:
            save_path = f"best_models/{self.model_pipeline.get_name()}_{best_model_name}_{model_name}"
            joblib.dump(self.model_pipeline.model, save_path)
       

        toc = time.perf_counter() # End Time
        print(f"Finished training with params. Took {(toc-tic):.2f}s.")
        return self.model_pipeline.model, train_dfs, eval_dfs, save_path

    def load_model_eval(self, df_train, model_name, save_path, model_type, shap_data_size=0.01):
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair for feature eval
        last_train_dfs, last_eval_dfs = train_dfs[-1], eval_dfs[-1]
        model = None
        if model_type=="mlp":
            model = keras.models.load_model(save_path)
        else:
            model = joblib.load(save_path)
        # self.plot_feature_importance(model, last_eval_dfs, model_name)
        # self.plot_shap(model, last_train_dfs, last_eval_dfs, model_name, shap_data_size)

        return model, train_dfs, eval_dfs

