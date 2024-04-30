class MLUtils:
    @staticmethod
    def create_XY(df, target_col_name='target'):
        features = [c for c in df.columns if c != target_col_name]
        x = df[features]
        y = df[target_col_name]
        return x, y

class ModelLogger:
    def log(self, msg):
        pass
    
    def reset(self):
        pass

class BasicModelLogger(ModelLogger):
    def __init__(self, msg_prefix):
        self.msg_prefix = msg_prefix
        # self.log_idx = 0
    
    def log(self, msg):
        print(f"{self.msg_prefix} - {msg}")
        # self.log_idx += 1
    
    def reset(self):
        # self.log_idx = 0
        pass

class NoopModelLogger(ModelLogger):
    def log(self, msg):
        pass
    
    def reset(self):
        pass
