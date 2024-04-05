from load_data import load_data_from_csv
from data_preprocessor.data_preprocessor import CompositeDataPreprocessor, ReduceMemUsageDataPreprocessor, FillNaPreProcessor
from data_preprocessor.feature_engineering import BasicFeaturesPreprocessor, DupletsTripletsPreprocessor, MovingAvgPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor, DTWKMeansPreprocessor
from data_preprocessor.polynomial_features import PolynomialFeaturesPreProcessor
from data_preprocessor.stockid_features import StockIdFeaturesPreProcessor
from data_generator.data_generator import DefaultTrainEvalDataGenerator, ManualKFoldDataGenerator, TimeSeriesKFoldDataGenerator

from model_pipeline.lgb_pipeline import LGBModelPipelineFactory
from model_pipeline.xgb_pipeline import XGBModelPipelineFactory
from model_pipeline.cbt_pipeline import CatBoostModelPipelineFactory

from model_post_processor.model_post_processor import CompositeModelPostProcessor, SaveModelPostProcessor

from train_pipeline.train_pipeline import DefaultTrainPipeline
from train_pipeline.train_optuna_pipeline import DefaultOptunaTrainPipeline

from train_pipeline.train_pipeline_callbacks import MAECallback
from utils.scoring_utils import ScoringUtils
from model_pipeline.dummy_models import BaselineEstimator

import optuna.integration.lightgbm as lgb
import optuna

import numpy as np

import sys

# model_name = sys.argv[1]
model_name = "best_model_2023_03_24"

# print("Model name is", sys.argv[1])

N_fold = 5
model_save_dir = './models/'

processors = [    
    ReduceMemUsageDataPreprocessor(verbose=True),
    # BasicFeaturesPreprocessor(),
    # DupletsTripletsPreprocessor(),
    # MovingAvgPreProcessor("wap"), 
    # StockIdFeaturesPreProcessor(),  
    # DTWKMeansPreprocessor(),    
    DropTargetNADataPreprocessor(),    
    RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id']),
    # FillNaPreProcessor(),
    # PolynomialFeaturesPreProcessor(),
]

test_processors = [
    BasicFeaturesPreprocessor(),
    # DupletsTripletsPreprocessor()
    MovingAvgPreProcessor("wap"),
    # DropTargetNADataPreprocessor(),
    # RemoveIrrelevantFeaturesDataPreprocessor(['date_id','time_id', 'row_id'])
    RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id'])
]
processor = CompositeDataPreprocessor(processors)
test_processors = CompositeDataPreprocessor(test_processors)

# DATA_PATH = '/kaggle/input'
DATA_PATH = '..'
df_train, df_test, revealed_targets, sample_submission = load_data_from_csv(DATA_PATH)
print(df_train.columns)

raw_data = df_train
# df_train = df_train[:1000]
df_train = processor.apply(df_train)
# df_test = test_processors.apply(df_test)
print(df_train.shape[0])
print(df_train.columns)


default_data_generator = DefaultTrainEvalDataGenerator()
k_fold_data_generator = ManualKFoldDataGenerator(n_fold=N_fold)
time_series_k_fold_data_generator = TimeSeriesKFoldDataGenerator(n_fold=N_fold, test_set_ratio=0.1)

model_post_processor = CompositeModelPostProcessor([
    SaveModelPostProcessor(save_dir=model_save_dir)
])

# lgb_pipeline = DefaultTrainPipeline(LGBModelPipelineFactory(), k_fold_data_generator, model_post_processor, [MAECallback()])
optuna_lgb_pipeline = DefaultOptunaTrainPipeline(LGBModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])
# optuna_lgb_pipeline = DefaultOptunaTrainPipeline(XGBModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])
# optuna_lgb_pipeline = DefaultOptunaTrainPipeline(CatBoostModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])


# hyper parameter tunning with optuna
best_param = optuna_lgb_pipeline.train(df_train)

# train model with param
# lgb_models, lgb_train_dfs, lgb_eval_dfs = optuna_lgb_pipeline.train_with_param(
#     df_train,
#     params={'n_estimators': 2700, 'reg_alpha': 1.666271247059715, 'reg_lambda': 0.0013314248446567097, 'colsample_bytree': 0.6512412430910787, 'subsample': 0.5550654570575708, 'learning_rate': 0.0124880163018859, 'max_depth': 11, 'num_leaves': 354, 'min_child_samples': 71,
#             'objective': 'regression_l1', 'random_state': 42, 'force_col_wise': True, "verbosity": -1}
# )
lgb_models, lgb_train_dfs, lgb_eval_df, best_model_name = optuna_lgb_pipeline.train_with_param(
    df_train,
    params=best_param,
    name = model_name
)

# load and eval model
lgb_models, lgb_train_dfs, lgb_eval_dfs = optuna_lgb_pipeline.load_model_eval(
    df_train,
    model_name,
    best_model_name
)


lgb_avg_mae = ScoringUtils.calculate_mae([lgb_models], lgb_eval_dfs)
print(lgb_avg_mae)

baseline_avg_mae = ScoringUtils.calculate_mae([BaselineEstimator()], [df_train])
print(baseline_avg_mae)
