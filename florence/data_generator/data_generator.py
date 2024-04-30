from collections import deque
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from utils.number_utils import NumberUtils
from utils.dataframe_utils import get_df_summary_str


class TrainEvalDataGenerator:
    def generate(self, df_train):
        return None, None, None

    def has_transformations(self):
        return False
    
class DefaultTrainEvalDataGenerator(TrainEvalDataGenerator):
    def generate(self, df_train):
        # whole df_train is training set, no validation set
        return [df_train], [None], 1

class ManualKFoldDataGenerator(TrainEvalDataGenerator):
    def __init__(self, n_fold=5):
        super().__init__()
        self.n_fold = n_fold
    
    def generate(self, df_train):
        train_dfs, eval_dfs = [], []
        index = np.arange(len(df_train))
        for fold in range(self.n_fold):
            fold_df_train = df_train[index%self.n_fold!=fold]
            fold_df_eval = df_train[index%self.n_fold==fold]
            train_dfs.append(fold_df_train)
            eval_dfs.append(fold_df_eval)
        return train_dfs, eval_dfs, self.n_fold


class TimeSeriesKFoldDataGenerator(TrainEvalDataGenerator):
    def __init__(self, n_fold=5, test_set_ratio=None, transform_pipeline=None):
        super().__init__()
        self.n_fold = n_fold
        self.test_set_ratio = test_set_ratio
        self.transform_pipeline = transform_pipeline

    def generate(self, data):
        train_dfs, eval_dfs = [], []
        if self.test_set_ratio:
            tscv = TimeSeriesSplit(n_splits=self.n_fold, test_size=int(len(data) * self.test_set_ratio))
        else:
            tscv = TimeSeriesSplit(n_splits=self.n_fold)
        for i, (train_index, test_index) in enumerate(tscv.split(data)):
            fold_df_train = data.iloc[train_index].copy(deep=True)
            fold_df_eval = data.iloc[test_index].copy(deep=True)

            print(f"TimeSeriesKFoldDataGenerator - before transformations (fold {i}) - fold_df_train: {get_df_summary_str(fold_df_train)}, fold_df_eval: {get_df_summary_str(fold_df_eval)}")
            if self.has_transformations():
                fold_df_train = self.transform_pipeline.fit_transform(fold_df_train)
                print(f"TimeSeriesKFoldDataGenerator - has_transformations (fold {i}) - fit_transform fold_df_train")
                fold_df_eval = self.transform_pipeline.transform(fold_df_eval)
                print(f"TimeSeriesKFoldDataGenerator - has_transformations (fold {i}) - transform fold_df_eval")
            else:
                print(f"TimeSeriesKFoldDataGenerator - has_transformations (fold {i}) = false")
            print(f"TimeSeriesKFoldDataGenerator - final (fold {i}) - fold_df_train: {get_df_summary_str(fold_df_train)}, fold_df_eval: {get_df_summary_str(fold_df_eval)}")
            
            train_dfs.append(fold_df_train)
            eval_dfs.append(fold_df_eval)
        return train_dfs, eval_dfs, self.n_fold

    def has_transformations(self):
        return self.transform_pipeline is not None


class TimeSeriesKFoldDataGeneratorOptimized(TrainEvalDataGenerator):
    def __init__(self, n_fold=5, test_set_ratio=None):
        super().__init__()
        self.n_fold = n_fold
        self.test_set_ratio = test_set_ratio

    def generate(self, data):
        train_dfs, eval_dfs = [], []
        if self.test_set_ratio:
            tscv = TimeSeriesSplit(n_splits=self.n_fold, test_size=int(len(data) * self.test_set_ratio))
        else:
            tscv = TimeSeriesSplit(n_splits=self.n_fold)
        for i, (train_index, test_index) in enumerate(tscv.split(data)):
            fold_df_train = data.iloc[train_index].copy(deep=True)
            fold_df_eval = data.iloc[test_index].copy(deep=True)
            yield fold_df_train, fold_df_eval
    

class TimeSeriesLastFoldDataGenerator(TrainEvalDataGenerator):

    def __init__(self, test_set_ratio=0.1, use_optimized_last_fold=False, transform_pipeline=None):
        super().__init__()
        self.test_set_ratio = test_set_ratio
        self.use_optimized_last_fold = use_optimized_last_fold
        self.transform_pipeline = transform_pipeline

    def generate(self, df_train):
        train = None
        eval = None
        
        if not self.use_optimized_last_fold:
            print(f"TimeSeriesLastFoldDataGenerator - generate (not optimized) - start - df_train: {get_df_summary_str(df_train)}")
            time_series_k_fold_data_generator = TimeSeriesKFoldDataGenerator(n_fold=2, test_set_ratio = 0.1)
            train_dfs, eval_dfs, num_train_eval_sets = time_series_k_fold_data_generator.generate(df_train)
            print(f"TimeSeriesLastFoldDataGenerator - generate (not optimized) - end")
    
            train = train_dfs[-1]
            eval = eval_dfs[-1]
            print(f"TimeSeriesLastFoldDataGenerator - extracted last fold (not optimized) - train: {get_df_summary_str(train)}, eval: {get_df_summary_str(eval)}")
        else:
            print(f"TimeSeriesLastFoldDataGenerator - generate - start (optimized) - df_train: {get_df_summary_str(df_train)}")
            time_series_k_fold_data_generator = TimeSeriesKFoldDataGeneratorOptimized(n_fold=2, test_set_ratio = 0.1)
            k_fold_iterator = time_series_k_fold_data_generator.generate(df_train)
            dd = deque(k_fold_iterator, maxlen=1)
            last_element = dd.pop()
            train = last_element[0]
            eval = last_element[1]
            print(f"TimeSeriesLastFoldDataGenerator - extracted last fold (optimized) - train: {get_df_summary_str(train)}, eval: {get_df_summary_str(eval)}")

        assert train is not None and eval is not None

        # if self.normalize:
        #     # NumberUtils.normalize_data(train)
        #     # print(f"TimeSeriesLastFoldDataGenerator - normalized train - memory: {train.memory_usage(index=True).sum() / 1024 / 1024}")
        #     # NumberUtils.normalize_data(eval)
        #     # print(f"TimeSeriesLastFoldDataGenerator - normalized eval - memory: {eval.memory_usage(index=True).sum() / 1024 / 1024}")
        #     normalize_columns = set([
        #         "imbalance_size",
        #         "matched_size",
        #         "bid_size",
        #         "ask_size",
        #     ])
        #     normalize_columns = list(normalize_columns.intersection(set(train.columns)))

        #     scaler = preprocessing.StandardScaler()
        #     scaler.fit(train[normalize_columns])
        #     train[normalize_columns] = scaler.transform(train[normalize_columns])
        #     print(f"TimeSeriesLastFoldDataGenerator - normalized train - memory: {get_df_memory_usage_mb(train)")
        #     eval[normalize_columns] = scaler.transform(eval[normalize_columns])
        #     print(f"TimeSeriesLastFoldDataGenerator - normalized eval - memory: {get_df_memory_usage_mb(eval)")

        if self.has_transformations():
            train = self.transform_pipeline.fit_transform(train)
            print(f"TimeSeriesLastFoldDataGenerator - has_transformations - fit_transform train")
            eval = self.transform_pipeline.transform(eval)
            print(f"TimeSeriesLastFoldDataGenerator - has_transformations - transform eval")
        else:
            print(f"TimeSeriesLastFoldDataGenerator - has_transformations = false")

        print(f"TimeSeriesLastFoldDataGenerator - final - train: {get_df_summary_str(train)}, eval: {get_df_summary_str(eval)}")

        return [train], [eval], 1

    def has_transformations(self):
        return self.transform_pipeline is not None
