from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

import numpy as np

from load_data import load_data_from_csv
import joblib

from data_preprocessor.data_preprocessor import CompositeDataPreprocessor, ReduceMemUsageDataPreprocessor, FillNaPreProcessor
from data_preprocessor.feature_engineering import BasicFeaturesPreprocessor, DupletsTripletsPreprocessor, MovingAvgPreProcessor, EWMAPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor, DTWKMeansPreprocessor, RemoveRecordsByStockDateIdPreprocessor, FarNearPriceFillNaPreprocessor, MovingAvgFillNaPreprocessor

from model_pipeline.model_pipeline import ModelPipeline

import os
import time

import matplotlib.pyplot as plt

from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, cross_validate
# os.environ["KERAS_BACKEND"] = "tensorflow"
# from keras_core.models import load_model
from keras.models import load_model

test_processors = [
    BasicFeaturesPreprocessor(),
    # DupletsTripletsPreprocessor()
    MovingAvgPreProcessor("wap"),
    DropTargetNADataPreprocessor(),
    # RemoveIrrelevantFeaturesDataPreprocessor(['date_id','time_id', 'row_id'])
    RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id'])
]
# test_processors = CompositeDataPreprocessor(test_processors)
# DATA_PATH = '/kaggle/input'


def prep_data():
    DATA_PATH = '..'
    df_train, df_val, df_test, revealed_targets, sample_submission = load_data_from_csv(DATA_PATH)

    processors = [
        ReduceMemUsageDataPreprocessor(verbose=True),
        RemoveRecordsByStockDateIdPreprocessor([
            {"stock_id": 19, "date_id": 438},
            {"stock_id": 101, "date_id": 328},
            {"stock_id": 131, "date_id": 35},
            {"stock_id": 158, "date_id": 388},
        ]),
        FarNearPriceFillNaPreprocessor(),
        # BasicFeaturesPreprocessor(),
        # DupletsTripletsPreprocessor(),
        MovingAvgPreProcessor("wap"),
        MovingAvgFillNaPreprocessor("wap", 1.0),
        # StockIdFeaturesPreProcessor(),
        # DTWKMeansPreprocessor(),
        # DfsPreProcessor(),
        # DropTargetNADataPreprocessor(),
        RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id', 'time_id', 'row_id']),
        FillNaPreProcessor(0.0),
        # PolynomialFeaturesPreProcessor(),
    ]

    processor = CompositeDataPreprocessor(processors)

    df_train = processor.apply(df_train)
    print(f"run pre-processors - applied on df_train")
    df_val = processor.apply(df_val)

    model_pipeline = ModelPipeline()
    X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(df_train, df_val)
    return X_train_fold, y_train_fold, X_val_fold, y_val_fold


def train_ensemble_model(estimators, X_train_fold, y_train_fold,
                         X_val_fold, y_val_fold):
    # lgb_model = joblib.load(
    #     'Best_model_pool/best_model_learning_rate_0.08210417223377245_n_estimators_2900_20240301_raw')
    #
    # mlp_model = load_model('ensemble_models/mlp_None_20240423_mlp_moving_avg.keras')

    final_estimator = GradientBoostingRegressor(
        n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
        random_state=42)
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv="prefit"
    )
    stacking_regressor = stacking_regressor.fit(X_train_fold, y_train_fold)
    regressor_score = stacking_regressor.score(X_val_fold, y_val_fold)
    return stacking_regressor, regressor_score

# X_train_fold, y_train_fold, X_val_fold, y_val_fold = prep_data()


def eval_ensemble_model(estimators, stacking_regressor, X_val_fold, y_val_fold):
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    for ax, (name, est) in zip(
        axs, estimators + [("Stacking Regressor", stacking_regressor)]
    ):
        scorers = {"R2": "r2", "MAE": "neg_mean_absolute_error"}

        start_time = time.time()
        scores = cross_validate(
            est, X_val_fold, y_val_fold, scoring=list(scorers.values()), n_jobs=-1, verbose=0
        )


        y_pred = cross_val_predict(est, X_val_fold, y_val_fold, n_jobs=-1, verbose=0)
        scores = {
            key: (
                f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- "
                f"{np.std(scores[f'test_{value}']):.2f}"
            )
            for key, value in scorers.items()
        }
        # y_pred = est.predict(X_val_fold, y_val_fold, n_jobs=-1, verbose=0)
        elapsed_time = time.time() - start_time

        display = PredictionErrorDisplay.from_predictions(
            y_true=y_val_fold,
            y_pred=y_pred,
            kind="actual_vs_predicted",
            ax=ax,
            scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
            line_kwargs={"color": "tab:red"},
        )
        ax.set_title(f"{name}\nEvaluation in {elapsed_time:.2f} seconds")

        for name, score in scores.items():
            ax.plot([], [], " ", label=f"{name}: {score}")
        ax.legend(loc="upper left")

    plt.suptitle("Single predictors versus stacked predictors")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'../img/ensemble_chart.jpg')
