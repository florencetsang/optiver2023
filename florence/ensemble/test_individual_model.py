import joblib
from os.path import isfile, join
from ensemble.util import train_ensemble_model, prep_data, eval_ensemble_model, lgb_features, xgb_features, cat_features
from utils.scoring_utils import ScoringUtils
from load_data import load_data_from_csv
from ensemble.util import (train_ensemble_model, prep_data, eval_ensemble_model, lgb_features, xgb_features, cat_features,
                           mlp_1_layer_features)
from data_preprocessor.data_preprocessor import CompositeDataPreprocessor, ReduceMemUsageDataPreprocessor, FillNaPreProcessor
from data_preprocessor.feature_engineering import BasicFeaturesPreprocessor, DupletsTripletsPreprocessor, MovingAvgPreProcessor, EWMAPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor, DTWKMeansPreprocessor, RemoveRecordsByStockDateIdPreprocessor, FarNearPriceFillNaPreprocessor, MovingAvgFillNaPreprocessor

from scikeras.wrappers import KerasRegressor
from keras.models import load_model
from model_pipeline.model_pipeline import ModelPipeline
from sklearn.pipeline import make_pipeline
from data_preprocessor.normalization import NormalizationDataTransformer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# all_features = [lgb_features, xgb_features, cat_features]
all_features = [mlp_1_layer_features]
base_path_of_model_pool = './ensemble/'


def feature_check(path_prefix: str, model_path, features):
    try:
        if path_prefix.startswith('mlp'):
            cur_model = KerasRegressor(load_model(join(base_path_of_model_pool, path_prefix, model_path)))
            cur_model.initialize(pd.DataFrame(np.zeros((3, cur_model.model.input_shape[1]))), y_train_fold.iloc[:3])
            feature_no = cur_model.model.input_shape[1]

        else:
            cur_model = joblib.load(join(base_path_of_model_pool, path_prefix, model_path))
            if path_prefix == 'lgb_pool':
                feature_no = len(cur_model.feature_name_)
            elif path_prefix == 'xgb_pool':
                feature_no = len(cur_model.feature_importances_)
            elif path_prefix == 'cb_pool':
                feature_no = len(cur_model.feature_names_)
        print(f"{str(model_path)}: model_feature_no: {feature_no}, input_feature: {len(features)}")

        if 'target' not in features:
            features += ['target']
        model_avg_mae = ScoringUtils.calculate_mae([cur_model], [df_val[features]])
        print(f"model: {model_path}, model_avg_mae: {model_avg_mae}")
        return cur_model
    except:
        print(f"Model: {model_path} num mismatch")

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
    BasicFeaturesPreprocessor(),
    DupletsTripletsPreprocessor(),
    MovingAvgPreProcessor("wap"),
    MovingAvgFillNaPreprocessor("wap", 1.0),
    # StockIdFeaturesPreProcessor(),
    # DTWKMeansPreprocessor(),
    # DfsPreProcessor(),
    # DropTargetNADataPreprocessor(),
    RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id', 'time_id', 'row_id']),
    FillNaPreProcessor(1.0),
    # PolynomialFeaturesPreProcessor(),
]

processor = CompositeDataPreprocessor(processors)

last_fold_data_generator_transform_pipeline = make_pipeline(
    # FeatureToolsDFSTransformer(
    #     group_by_stock=True,
    #     group_by_date=False,
    #     group_by_stock_date=False,
    # ),
    # StockIdFeaturesDataTransformer(),
    NormalizationDataTransformer(
        [
            "imbalance_size",
            "matched_size",
            "bid_size",
            "ask_size",
        ],
        "closing_movements",
    ),
    # RemoveIrrelevantFeaturesDataTransformer(['stock_id', 'date_id','time_id', 'row_id', "stock_date_id"]),
    verbose=True,
)

df_train, df_val = df_train[:10000], df_val[:10000]
normalization_pipeline = last_fold_data_generator_transform_pipeline.fit(df_train)

df_train = processor.apply(df_train)
df_train = normalization_pipeline.transform(df_train)
df_val = processor.apply(df_val)
df_val = normalization_pipeline.transform(df_val)


model_pipeline = ModelPipeline()
X_train_fold, y_train_fold, X_val_fold, y_val_fold = model_pipeline.create_XY(df_train, df_val)


# for model_features in all_features:
#     for model_path, features in model_features.items():
#         if str(model_path).startswith('lgb'):
#             path_prefix = 'lgb_pool'
#         elif str(model_path).startswith('xgb'):
#             path_prefix = 'xgb_pool'
#         elif str(model_path).startswith('xgb'):
#             path_prefix = 'cb_pool'
#         elif str(model_path).startswith('mlp'):
#             path_prefix = 'mlp_1layer_pool'
#         feature_check(path_prefix, model_path, features)
#     print("-------------------------------------------------------------")

cur_model = joblib.load("ensemble/Best_ensemble_model/with_all_data_mlp_2")
from sklearn.inspection import permutation_importance

result = permutation_importance(
    cur_model, X_val_fold, y_val_fold, n_repeats=10, random_state=42, n_jobs=2
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X_train_fold.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.savefig(f'img/permutation_importance_stack_model_2.jpg')
feature_importance_img = plt.imread(f"img/permutation_importance_stack_model_2.jpg")
plt.imshow(feature_importance_img)
