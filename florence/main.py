from load_data import load_data_from_csv
from data_preprocessor.data_preprocessor import CompositeDataPreprocessor
from data_preprocessor.feature_engineering import EnrichDFDataPreprocessor, MovingAvgPreProcessor, RemoveIrrelevantFeaturesDataPreprocessor, DropTargetNADataPreprocessor
from data_generator.data_generator import DefaultTrainEvalDataGenerator, ManualKFoldDataGenerator, TimeSeriesKFoldDataGenerator

from model_pipeline.lgb_pipeline import LGBModelPipelineFactory

from model_post_processor.model_post_processor import CompositeModelPostProcessor, SaveModelPostProcessor

from train_pipeline.train_pipeline import DefaultTrainPipeline

from train_pipeline.train_pipeline_callbacks import MAECallback
from utils.scoring_utils import ScoringUtils
from model_pipeline.dummy_models import BaselineEstimator


N_fold = 5
model_save_dir = './models/'


processors = [
    EnrichDFDataPreprocessor(),
    MovingAvgPreProcessor("wap"),
    DropTargetNADataPreprocessor(),
    RemoveIrrelevantFeaturesDataPreprocessor(['stock_id', 'date_id','time_id', 'row_id'])
]
processor = CompositeDataPreprocessor(processors)

# DATA_PATH = '/kaggle/input'
DATA_PATH = '..'
df_train, df_test, revealed_targets, sample_submission = load_data_from_csv(DATA_PATH)
print(df_train.columns)

df_train = processor.apply(df_train)
print(df_train.shape[0])
print(df_train.columns)
# display(df_train.tail())






default_data_generator = DefaultTrainEvalDataGenerator()
k_fold_data_generator = ManualKFoldDataGenerator(n_fold=N_fold)
time_series_k_fold_data_generator = TimeSeriesKFoldDataGenerator(n_fold=N_fold)

model_post_processor = CompositeModelPostProcessor([
    SaveModelPostProcessor(save_dir=model_save_dir)
])

# lgb_pipeline = DefaultTrainPipeline(LGBModelPipelineFactory(), k_fold_data_generator, model_post_processor, [MAECallback()])
lgb_pipeline = DefaultTrainPipeline(LGBModelPipelineFactory(), time_series_k_fold_data_generator, model_post_processor, [MAECallback()])

lgb_models, lgb_model_res, lgb_train_dfs, lgb_eval_dfs, lgb_num_train_eval_sets, lgb_callback_results = lgb_pipeline.train(df_train)

lgb_avg_mae = ScoringUtils.calculate_mae(lgb_models, lgb_eval_dfs)
print(lgb_avg_mae)

baseline_avg_mae = ScoringUtils.calculate_mae([BaselineEstimator()], [df_train])
print(baseline_avg_mae)
