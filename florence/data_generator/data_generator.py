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
    def __init__(self, n_fold=5, test_size=2):
        super().__init__()
        self.n_fold = n_fold
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=self.n_fold, test_size=self.test_size, gap=0)

    def generate(self, df_train):
        train_dfs, eval_dfs = [], []
        for i, (train_index, test_index) in enumerate(self.tscv.split(df_train)):
            fold_df_train = df_train[train_index]
            fold_df_eval = df_train[test_index]
            train_dfs.append(fold_df_train)
            eval_dfs.append(fold_df_eval)
        return train_dfs, eval_dfs, self.n_fold
