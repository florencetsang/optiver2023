class MLUtils:
    @staticmethod
    def create_XY(df, target_col_name='target'):
        features = [c for c in df.columns if c != target_col_name]
        x = df[features].values
        y = df[target_col_name].values
        return x, y
