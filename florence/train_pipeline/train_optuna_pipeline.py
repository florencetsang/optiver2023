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

class DefaultOptunaTrainPipeline():
    def __init__(
            self,
            model_pipeline_factory: ModelPipelineFactory,
            train_eval_data_generator: TrainEvalDataGenerator,
            model_post_processor: ModelPostProcessor,
            callbacks
    ):
        self.model_pipeline_factory = model_pipeline_factory
        self.train_eval_data_generator = train_eval_data_generator
        self.model_post_processor = model_post_processor
        self.callbacks = callbacks

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

    def plot_shap(self, lgbm_model, df_train, df_eval, save_name):
        # shap
        # Create a SHAP explainer object for the LightGBM model
        explainer = shap.TreeExplainer(lgbm_model)

        # Sample 10% of the test data to use for faster computation of SHAP values
        model_pipeline = self.model_pipeline_factory.create_model_pipeline()
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(df_train, df_eval)
        sample = pd.DataFrame(X_val_fold).sample(frac=0.0001, random_state=42)

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
            for fold_index in range(len(train_dfs)):
                # Split the data into training and validation sets
                model_pipeline = self.model_pipeline_factory.create_model_pipeline()
                X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(train_dfs[fold_index], eval_dfs[fold_index])

                # create the LightGBM regressor with the optimized parameters
                model = lgb.LGBMRegressor(**param)
                # Train the model on the training set
                if early_stopping_flag:
                    eval_res = {}
                    # Use early stopping if enabled
                    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], eval_metric='l1',
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=100),
                                  lgb.record_evaluation(eval_res)
                              ])
                else:
                    model.fit(X_train_fold, y_train_fold)

                # Make predictions on the validation set and calculate the MSE score
                y_pred = model.predict(X_val_fold)
                mse = mean_absolute_error(y_val_fold, y_pred)
                mse_list.append(mse)

            # Return the trained model and the average MSE score
            return model, np.mean(mse_list)

        def objective(trial):
            # set up the parameters to be optimized
            param = {
                'objective': 'regression_l1',
                'random_state': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000, step=100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'num_leaves': trial.suggest_int('num_leaves', 31, 512),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                # 'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
                'force_col_wise': True,
                "verbosity": -1,
            }

            # perform cross-validation using the optimized LightGBM regressor
            model, mean_score = cross_validation_fcn(train_dfs, eval_dfs, param, early_stopping_flag=True)

            # retrieve the best iteration of the model and store it as a user attribute in the trial object
            best_iteration = model.best_iteration_
            trial.set_user_attr('best_iteration', best_iteration)

            return mean_score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)

        print("Number of finished trials: {}".format(len(study.trials)))

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

        best_param = {
            'objective': 'regression_l1',
            'random_state': 42,
            'force_col_wise': True,
            "verbosity": -1,
        }

        best_param.update(best_trial.params)

        return best_param

    def train_with_param(self, df_train, params, name):
        print(f"Start training with params: {params}.")
        tic = time.perf_counter() # Start Time
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair
        train_dfs, eval_dfs = train_dfs[-1], eval_dfs[-1]

        # Fit the model to the training data
        lgbm_model = lgb.LGBMRegressor(**params)
        model_pipeline = self.model_pipeline_factory.create_model_pipeline()
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(train_dfs, eval_dfs)
        lgbm_model.fit(X_train_fold, y_train_fold)
        best_model_name = f"best_models/best_model_learning_rate_{params['learning_rate']}_n_estimators_{params['n_estimators']}_{name}"
        joblib.dump(lgbm_model, best_model_name)
        toc = time.perf_counter() # End Time
        print(f"Finished training with params. Took {(toc-tic):.2f}s.")
        return lgbm_model, train_dfs, eval_dfs, best_model_name

    def load_model_eval(self, df_train, model_name, best_model_name):
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair for feature eval
        last_train_dfs, last_eval_dfs = train_dfs[-1], eval_dfs[-1]
        lgbm_model = joblib.load(f"best_models/{best_model_name}")
        self.plot_feature_importance(lgbm_model, last_eval_dfs, model_name)
        self.plot_shap(lgbm_model, last_train_dfs, last_eval_dfs, model_name)

        return lgbm_model, train_dfs, eval_dfs

