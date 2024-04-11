import numpy as np
from sklearn.model_selection import TimeSeriesSplit

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
