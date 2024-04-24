import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from utils.number_utils import NumberUtils


class TrainEvalDataGenerator:
    def generate(self, df_train):
        return None, None, None
    
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
            train_dfs.append(fold_df_train)
            eval_dfs.append(fold_df_eval)
        return train_dfs, eval_dfs, self.n_fold
    
class TimeSeriesLastFoldDataGenerator(TrainEvalDataGenerator):

    def __init__(self, test_set_ratio=0.1, normalize=False):
        super().__init__()
        self.test_set_ratio = test_set_ratio
        self.normalize = normalize

    def generate(self, df_train):    
        time_series_k_fold_data_generator = TimeSeriesKFoldDataGenerator(n_fold=2, test_set_ratio = 0.1)
        train_dfs, eval_dfs, num_train_eval_sets = time_series_k_fold_data_generator.generate(df_train)

        train = train_dfs[-1]
        eval = eval_dfs[-1]

        if self.normalize:
            NumberUtils.normalize_data(train)
            NumberUtils.normalize_data(eval)
            # normalize_columns = set([
            #     "imbalance_size",
            #     "matched_size",
            #     "bid_size",
            #     "ask_size",
            # ])
            # normalize_columns = list(normalize_columns.intersection(set(train.columns)))

            # scaler = preprocessing.StandardScaler()
            # scaler.fit(train[normalize_columns])
            # train[normalize_columns] = scaler.transform(train[normalize_columns])
            # eval[normalize_columns] = scaler.transform(eval[normalize_columns])

        return [train], [eval], 1
    

