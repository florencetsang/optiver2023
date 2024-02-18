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

    def plot_feature_importance(self, lgbm_model, df_train):
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
        plt.savefig('img/feature_importance.jpg')
        plt.show()

    def plot_shap(self, lgbm_model, df_train, df_eval):
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
        plt.savefig('img/shap.jpg')
        plt.show()

    def train(self, df_train):
        print(f"generate data")
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)

        models = []
        model_res = []

        print(f"start training, num_train_eval_sets: {num_train_eval_sets}")

        def cross_validation_fcn(train_dfs, eval_dfs, model, early_stopping_flag=False):
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
                # X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                # y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

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

            # create the LightGBM regressor with the optimized parameters
            model = lgb.LGBMRegressor(**param)

            # perform cross-validation using the optimized LightGBM regressor
            model, mean_score = cross_validation_fcn(train_dfs, eval_dfs, model, early_stopping_flag=True)

            # retrieve the best iteration of the model and store it as a user attribute in the trial object
            best_iteration = model.best_iteration_
            trial.set_user_attr('best_iteration', best_iteration)

            return mean_score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        best_trial = study.best_trial

        print("  Value: {}".format(best_trial.value))

        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))


        def train_with_best_param(best_trial):
            # train with best param
            # hp_lgbm = {'reg_alpha': 0.05656361784137243, 'reg_lambda': 0.003463515502393046,
            #            'colsample_bytree': 0.9013882987314858, 'subsample': 0.7117252548723397,
            #            'learning_rate': 0.016272774278785776, 'max_depth': 7, 'num_leaves': 100, 'min_child_samples': 36,
            #            'cat_smooth': 84}
            hp_lgbm = best_trial.params.items()
            lgbm_model = lgb.LGBMRegressor(**hp_lgbm)

            # Fit the model to the training data
            model_pipeline = self.model_pipeline_factory.create_model_pipeline()
            X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(train_dfs[4], eval_dfs[4])
            lgbm_model.fit(X_train_fold, y_train_fold)

            # Use the trained model to make predictions on the test data
            y_pred_lgbm = lgbm_model.predict(X_val_fold)

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            print("Best rmse:", np.sqrt(mean_squared_error(y_pred_lgbm, y_val_fold)))

            # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
            print("R2 using LightGBM: ", r2_score(y_val_fold, y_pred_lgbm))
            return lgbm_model

        def plot_feature_importance(lgbm_model):
            # feature importance
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            import textwrap #pip install textwrap3
            # Set the feature names of the LightGBM model to be the columns of the encoded data frame (excluding the target column)
            lgbm_model.feature_names = train_dfs[0].drop('target', axis=1).columns

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
            plt.show()


        def plot_shap(lgbm_model):
            # shap
            # Create a SHAP explainer object for the LightGBM model
            explainer = shap.TreeExplainer(lgbm_model)

            # Sample 10% of the test data to use for faster computation of SHAP values
            model_pipeline = self.model_pipeline_factory.create_model_pipeline()
            X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(train_dfs[4], eval_dfs[4])
            sample = pd.DataFrame(X_val_fold).sample(frac=0.1, random_state=42)

            # Calculate the SHAP values for the sampled test data using the explainer
            shap_values = explainer.shap_values(sample)

            # Create a SHAP summary plot of the feature importances based on the SHAP values
            feature_names = train_dfs[0].drop('target', axis=1).columns
            shap.summary_plot(shap_values, sample, show=False, feature_names=feature_names)

            # Adjust the layout of the plot to avoid overlapping labels
            plt.tight_layout()

            # Show the plot
            plt.show()

        lgbm_model = train_with_best_param(best_trial)
        plot_feature_importance(lgbm_model)
        plot_shap(lgbm_model)

        # for fold in range(num_train_eval_sets):
        #     print(f"Training fold {fold} - start")
        #
        #     model_pipeline = self.model_pipeline_factory.create_model_pipeline()
        #     model_pipeline.init_model()
        #     print(f"Training fold {fold} - initialized")
        #
        #     fold_df_train = train_dfs[fold]
        #     fold_df_eval = eval_dfs[fold]
        #     print(f"Training fold {fold} - train size: {fold_df_train.shape}, eval size: {fold_df_eval.shape}")
        #
        #     print(f"Training fold {fold} - start training")
        #     train_res = model_pipeline.train(fold_df_train, fold_df_eval)
        #     fold_model = model_pipeline.get_model()
        #     models.append(fold_model)
        #     model_res.append(train_res)
        #     print(f"Training fold {fold} - finished training")
        #
        #     self.model_post_processor.process(fold_model, model_pipeline, fold)
        #     print(f"Training fold {fold} - finished post processing")
        #
        #     print(f"Training fold {fold} - end")
        
        print(f"finished training, num_train_eval_sets: {num_train_eval_sets}")

        callback_results = []
        for callback in self.callbacks:
            callback_res = callback.on_callback(models, model_res, train_dfs, eval_dfs, num_train_eval_sets)
            callback_results.append(callback_res)

        return models, model_res, train_dfs, eval_dfs, num_train_eval_sets, callback_results

    def train_with_param(self, df_train, params):
        def train_with_best_param(df_train, df_eval, params):
            lgbm_model = lgb.LGBMRegressor(**params)

            # Fit the model to the training data
            model_pipeline = self.model_pipeline_factory.create_model_pipeline()
            X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(df_train, df_eval)
            lgbm_model.fit(X_train_fold, y_train_fold)

            # Use the trained model to make predictions on the test data
            y_pred_lgbm = lgbm_model.predict(X_val_fold)

            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            print("Best mae:", mean_absolute_error(y_pred_lgbm, y_val_fold))

            # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
            print("R2 using LightGBM: ", r2_score(y_val_fold, y_pred_lgbm))
            return lgbm_model

        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair
        train_dfs, eval_dfs = train_dfs[-1], eval_dfs[-1]
        lgbm_model = train_with_best_param(train_dfs, eval_dfs, params)
        joblib.dump(lgbm_model, f"best_models/best_model_learning_rate_{params['learning_rate']}_n_estimators_{params['n_estimators']}")

        self.plot_feature_importance(lgbm_model, eval_dfs)
        self.plot_shap(lgbm_model, df_train, eval_dfs)

        return lgbm_model, train_dfs, eval_dfs

    def load_model_eval(self, df_train, path):
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)
        # pick last training testing pair
        train_dfs, eval_dfs = train_dfs[-1], eval_dfs[-1]
        lgbm_model = joblib.load(path)
        self.plot_feature_importance(lgbm_model, eval_dfs)
        self.plot_shap(lgbm_model, df_train, eval_dfs)

        return lgbm_model, train_dfs, eval_dfs

