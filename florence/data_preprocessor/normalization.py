from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing

class NormalizationDataTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, normalize_columns, normalize_columns_pattern):
        self.normalize_columns = normalize_columns
        self.normalize_columns_pattern = normalize_columns_pattern
        self.scaler = preprocessing.StandardScaler()
        self.final_columns = []

    def fit(self, df, y=None):
        final_columns = []
        for col in df.columns:
            if col in self.normalize_columns or col.find(self.normalize_columns_pattern) != -1:
                final_columns.append(col)
        self.final_columns = final_columns
        self.scaler.fit(df[self.final_columns])
        print(f"NormalizationDataTransformer - fit - final_columns: {self.final_columns}")
        return self

    def transform(self, df):
        # need to copy a new df, otherwise, it will modify the view and the underlying df
        final_df = df.copy(deep=True)
        final_df[self.final_columns] = self.scaler.transform(final_df[self.final_columns])
        print(f"NormalizationDataTransformer - transform - final_columns: {self.final_columns}")
        return final_df
