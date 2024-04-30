import argparse

from load_data import load_data_from_csv
from data_preprocessor.data_preprocessor import CompositeDataPreprocessor, ReduceMemUsageDataPreprocessor, FillNaPreProcessor

from data_preprocessor.feature_engineering import BasicFeaturesPreprocessor, DupletsTripletsPreprocessor, MovingAvgPreProcessor, EWMAPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor, DTWKMeansPreprocessor, RemoveRecordsByStockDateIdPreprocessor, FarNearPriceFillNaPreprocessor, MovingAvgFillNaPreprocessor, EWMAFillNaPreprocessor, RemoveIrrelevantFeaturesDataTransformer
from data_preprocessor.polynomial_features import PolynomialFeaturesPreProcessor
from data_preprocessor.stockid_features import StockIdFeaturesPreProcessor, StockIdFeaturesDataTransformer
from data_preprocessor.deep_feature_synthesis import DfsPreProcessor, StockDateIdPreprocessor, FeatureToolsDFSTransformer
from data_preprocessor.normalization import NormalizationDataTransformer
from data_preprocessor.stocks_pca_preprocessor import StocksPcaPreProcessor
from data_generator.data_generator import DefaultTrainEvalDataGenerator, ManualKFoldDataGenerator, TimeSeriesKFoldDataGenerator, TimeSeriesLastFoldDataGenerator

from model_pipeline.lgb_pipeline import LGBModelPipelineFactory
from model_pipeline.xgb_pipeline import XGBModelPipelineFactory
from model_pipeline.cbt_pipeline import CatBoostModelPipelineFactory
from model_pipeline.mlp_pipeline import MLPModelPipelineFactory

from model_post_processor.model_post_processor import CompositeModelPostProcessor, SaveModelPostProcessor

from train_pipeline.train_pipeline import DefaultTrainPipeline
from train_pipeline.train_optuna_pipeline import DefaultOptunaTrainPipeline

from train_pipeline.train_pipeline_callbacks import MAECallback
from utils.scoring_utils import ScoringUtils
from utils.number_utils import NumberUtils
from model_pipeline.dummy_models import BaselineEstimator

import optuna.integration.lightgbm as lgb
import optuna

import numpy as np
from sklearn.pipeline import make_pipeline

import sys

parser = argparse.ArgumentParser(prog='optuna_main', description='optuna_main')
parser.add_argument('model_name')
parser.add_argument('model_type')
parser.add_argument('--trials', type=int, default=10, help="number of optuna trials")

args = parser.parse_args()
model_name = args.model_name
model_type = args.model_type
num_trials = args.trials
print(f"Model name is {model_name}, Model type: {model_type}, num_trials: {num_trials}")

N_fold = 5
model_save_dir = './models/'
plot_path = "./img"

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
    # MovingAvgPreProcessor("wap"),
    # MovingAvgFillNaPreprocessor("wap", 1.0),
    # EWMAPreProcessor("wap", 10),
    # EWMAFillNaPreprocessor("wap", 1.0),
    # StockIdFeaturesPreProcessor(),   
    StocksPcaPreProcessor(),
    # DTWKMeansPreprocessor(),
    # DfsPreProcessor(),
    # StockDateIdPreprocessor(), 
    # FeatureToolsDFSPreprocessor(),
    # DropTargetNADataPreprocessor(),    
    # RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id']),
    # FillNaPreProcessor(1.0),
    # PolynomialFeaturesPreProcessor(),
]

processor = CompositeDataPreprocessor(processors)

# DATA_PATH = '/kaggle/input'
DATA_PATH = '..'
df_train, df_val, df_test, revealed_targets, sample_submission = load_data_from_csv(DATA_PATH)
print(f"df details - df_train: {df_train.shape}, df_val: {df_val.shape}")
print(df_train.columns)

# run data preprocessors on df_train and df_val
raw_data = df_train
print(f"run pre-processors - start")
df_train = processor.apply(df_train)
print(f"run pre-processors - applied on df_train")
df_val = processor.apply(df_val)
print(f"run pre-processors - applied on df_val")
print(f"df details - df_train: {df_train.shape}, df_val: {df_val.shape}")
print(df_train.columns)

default_data_generator = DefaultTrainEvalDataGenerator()
k_fold_data_generator = ManualKFoldDataGenerator(n_fold=N_fold)

# ---------- time series k fold - start ----------
# ---------- time series k fold - choose sklearn fit-transform pipeline or none pipeline ----------
time_series_k_fold_data_generator_transform_pipeline = make_pipeline(
    StockIdFeaturesDataTransformer(),
    RemoveIrrelevantFeaturesDataTransformer(['stock_id', 'date_id','time_id', 'row_id', "stock_date_id"]),
)
time_series_k_fold_data_generator_transform_pipeline = None
time_series_k_fold_data_generator = TimeSeriesKFoldDataGenerator(
    n_fold=N_fold,
    test_set_ratio=0.05,
    transform_pipeline=time_series_k_fold_data_generator_transform_pipeline
)
# ---------- time series k fold - end ----------

# ---------- time series last fold - start ----------
# ---------- time series last fold - choose sklearn fit-transform pipeline or none pipeline ----------
last_fold_data_generator_transform_pipeline = make_pipeline(
    # FeatureToolsDFSTransformer(
    #     group_by_stock=True,
    #     group_by_date=False,
    #     group_by_stock_date=False,
    # ),
    StockIdFeaturesDataTransformer(),
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
# last_fold_data_generator_transform_pipeline = None
last_fold_data_generator = TimeSeriesLastFoldDataGenerator(
    test_set_ratio=0.1,
    use_optimized_last_fold=True,
    transform_pipeline=last_fold_data_generator_transform_pipeline,
)
# ---------- time series last fold - end ----------

model_post_processor = CompositeModelPostProcessor([
    SaveModelPostProcessor(save_dir=model_save_dir)
])

# ---------- create optuna pipeline ----------
optuna_pipeline = None
if model_type == 'lgb':
    optuna_pipeline = DefaultOptunaTrainPipeline(LGBModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])
elif model_type == 'xgb':
    optuna_pipeline = DefaultOptunaTrainPipeline(XGBModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])
elif model_type == 'cb':
    optuna_pipeline = DefaultOptunaTrainPipeline(CatBoostModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])
elif model_type == 'mlp':
    optuna_pipeline = DefaultOptunaTrainPipeline(
        MLPModelPipelineFactory(
            model_name,
            plot_path,
            len(df_train.columns)-1
        ),
        last_fold_data_generator,
        model_post_processor,
        [MAECallback()],
        num_trials=num_trials,
    )


# hyper parameter tunning with optuna
best_param = optuna_pipeline.train(df_train)

# train model with param
# trained_models, train_dfs, eval_dfs = optuna_pipeline.train_with_param(
#     df_train,
#     params={'n_estimators': 2700, 'reg_alpha': 1.666271247059715, 'reg_lambda': 0.0013314248446567097, 'colsample_bytree': 0.6512412430910787, 'subsample': 0.5550654570575708, 'learning_rate': 0.0124880163018859, 'max_depth': 11, 'num_leaves': 354, 'min_child_samples': 71,
#             'objective': 'regression_l1', 'random_state': 42, 'force_col_wise': True, "verbosity": -1}
# )
trained_models, train_dfs, eval_dfs, save_path = optuna_pipeline.train_with_param(
    df_train,
    df_val,
    params=best_param,
    model_name = model_name,
    model_type = model_type
)

# load and eval model
# include transformed df_train and df_val (e.g. mlp data scaler, auto feature engineering)
trained_models, df_train, df_val = optuna_pipeline.load_model_eval(
    df_train,
    df_val,
    model_name,
    save_path,
    model_type = model_type
)

model_avg_mae = ScoringUtils.calculate_mae([trained_models], [df_val])
print(model_avg_mae)

baseline_avg_mae = ScoringUtils.calculate_mae([BaselineEstimator()], [df_val])
print(baseline_avg_mae)
