from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

import numpy as np

from load_data import load_data_from_csv
import joblib

from data_preprocessor.feature_engineering import BasicFeaturesPreprocessor, DupletsTripletsPreprocessor, MovingAvgPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor, DTWKMeansPreprocessor
from data_preprocessor.data_preprocessor import CompositeDataPreprocessor, ReduceMemUsageDataPreprocessor, FillNaPreProcessor



import os

# test_processors = [
#     BasicFeaturesPreprocessor(),
#     # DupletsTripletsPreprocessor()
#     MovingAvgPreProcessor("wap"),
#     # DropTargetNADataPreprocessor(),
#     # RemoveIrrelevantFeaturesDataPreprocessor(['date_id','time_id', 'row_id'])
#     RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id'])
# ]
# test_processors = CompositeDataPreprocessor(test_processors)
# DATA_PATH = '/kaggle/input'
DATA_PATH = '../..'
df_train, df_test, revealed_targets, sample_submission = load_data_from_csv(DATA_PATH)
# df_test = test_processors.apply(df_test)
lgb_model = joblib.load('../best_models/best_model_2023_02_19')

# lgb_model = joblib.load('Best_model_pool/best_model_learning_rate_0.08210417223377245_n_estimators_2900_20240301_raw')
# os.environ["KERAS_BACKEND"] = "tensorflow"
# from keras_core.models import load_model
from keras.models import load_model
mlp_model = load_model('Best_model_pool/mlp_4.keras')

estimators = [
    ('lgb', lgb_model),
    ('mlp', mlp_model),
]

final_estimator = GradientBoostingRegressor(
    n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
    random_state=42)
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator
)

import time

import matplotlib.pyplot as plt

from sklearn.metrics import PredictionErrorDisplay
from sklearn.model_selection import cross_val_predict, cross_validate

fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)

for ax, (name, est) in zip(
    axs, estimators + [("Stacking Regressor", stacking_regressor)]
):
    scorers = {"R2": "r2", "MAE": "neg_mean_absolute_error"}

    start_time = time.time()
    scores = cross_validate(
        est, X, y, scoring=list(scorers.values()), n_jobs=-1, verbose=0
    )
    elapsed_time = time.time() - start_time

    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)
    scores = {
        key: (
            f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- "
            f"{np.std(scores[f'test_{value}']):.2f}"
        )
        for key, value in scorers.items()
    }

    display = PredictionErrorDisplay.from_predictions(
        y_true=y,
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
plt.show()
