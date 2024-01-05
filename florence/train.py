import os
from feature_engineering import enrich_df_with_features
import lightgbm as lgb 
import xgboost as xgb 
import catboost as cbt 
import numpy as np
import joblib

nonfeatures = ['stock_id', 'date_id','time_id', 'row_id','target']

class DataPreprocessor:
    def apply(self, df):
        return df
    
class CompositeDataPreprocessor(DataPreprocessor):
    def __init__(self, processors):
        self.processors = processors

    def apply(self, df):
        processed_df = df
        for processor in self.processors:
            processed_df = processor.apply(processed_df)
        return processed_df

class RemoveIrrelevantFeaturesDataPreprocessor(DataPreprocessor):
    def __init__(self, non_features):
        super().__init__()
        self.non_features = non_features
    
    def apply(self, df):
        useful_features = [c for c in df.columns if c not in self.nonfeatures]
        processed_df = df[useful_features]
        return processed_df

class DropTargetNADataPreprocessor(DataPreprocessor):
    def __init__(self, target_col_name='target'):
        super().__init__()
        self.target_col_name = target_col_name
    
    def apply(self, df):
        processed_df = df.dropna(subset=[self.target_col_name])
        return processed_df

class MLUtils:
    @staticmethod
    def create_XY(df, target_col_name='target'):
        features = [c for c in df.columns if c != target_col_name]
        x = df[features].values
        y = df[target_col_name].values
        return x, y

class ModelPipeline:
    model = None

    def init_model(self):
        pass
    
    def train(self, df_train, df_eval):
        train_X, train_Y = MLUtils.create_XY(df_train)
        eval_X, eval_Y = MLUtils.create_XY(df_eval)
        self._train(train_X, train_Y, eval_X, eval_Y)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        pass

    def get_model(self):
        return self.model

    def get_name(self):
        return "AbstractModelPipeline"

class LGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
        self.model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=50)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        self.model.fit(train_X, train_Y, 
                eval_set=[(eval_X, eval_Y)], 
                callbacks=[lgb.early_stopping(100)]
            )
    
    def get_name(self):
        return "lgb"
    
class XGBModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
        self.model = xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500, early_stopping_rounds = 100)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        self.model.fit(train_X, train_Y, 
                eval_set=[(eval_X, eval_Y)], 
            )
        
    def get_name(self):
        return "xgb"

class CatBoostModelPipeline(ModelPipeline):
    def __init__(self):
        super().__init__()
        
    def init_model(self):
        self.model = cbt.CatBoostRegressor(objective='MAE', iterations=50)
    
    def _train(self, train_X, train_Y, eval_X, eval_Y):
        self.model.fit(train_X, train_Y, 
                eval_set=[(eval_X, eval_Y)], 
                early_stopping_rounds = 100
            )
        
    def get_name(self):
        return "cbt"

class ModelPipelineFactory:
    def create_model_pipeline(self) -> ModelPipeline:
        return None

class LGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return LGBModelPipeline()

class XGBModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return XGBModelPipeline()
    
class CatBoostModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return CatBoostModelPipeline()

class ManualKFoldModelPipeline(ModelPipeline):
    def __init__(self, model_pipeline_factory: ModelPipelineFactory, n_fold=5, dump_model=True):
        super().__init__()
        self.model_pipeline_factory = model_pipeline_factory
        self.n_fold = n_fold
        self.dump_model = dump_model
    
    def init_model(self):
        self.models = []
    
    def train(self, df_train, df_eval):
        # do not need to pass in df_eval
        index = np.arange(len(df_train))
        for fold in range(self.n_fold):
            print(f"Training fold {fold} - start")
            model_pipeline = self.model_pipeline_factory.create_model_pipeline()
            model_pipeline.init_model()
            print(f"Training fold {fold} - initialized")
            fold_df_train = df_train[index%self.n_fold!=fold]
            fold_df_eval = df_train[index%self.n_fold==fold]
            model_pipeline.train(fold_df_train, fold_df_eval)
            print(f"Training fold {fold} - finished training")
            fold_model = model_pipeline.get_model()
            self.models.append(fold_model)
            if self.dump_model and fold_model is not None:
                joblib.dump(fold_model, f'./models/{model_pipeline.get_name()}_{fold}.model')
            print(f"Training fold {fold} - end")
    
    def get_model(self):
        return self.models
    
    def get_name(self):
        return "ManualKFoldModelPipeline"


# to be run from a main.py or notebook
def test_refactor_main():
    processors = [
        RemoveIrrelevantFeaturesDataPreprocessor(),
        DropTargetNADataPreprocessor()
    ]
    processor = CompositeDataPreprocessor(processors)

    # TODO: get data
    data = None

    processed_data = processor.apply(data)

    model_pipeline = ManualKFoldModelPipeline(LGBModelPipelineFactory, n_fold=5, dump_model=True)
    model_pipeline.init_model()
    # specific set up for ManualKFoldModelPipeline to pass None
    model_pipeline.train(processed_data, None)
    models = model_pipeline.get_model()


model_dict = {
    'lgb': lgb.LGBMRegressor(objective='regression_l1', n_estimators=50),
    'xgb': xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500, early_stopping_rounds = 100),
    'cbt': cbt.CatBoostRegressor(objective='MAE', iterations=50),
}

def create_train_XY(df_train):

    df_train_ = enrich_df_with_features(df_train)    
    features = [c for c in df_train_.columns if c not in nonfeatures]
    X = df_train_.dropna(subset=['target'])[features].values
    Y = df_train_.dropna(subset=['target'])['target'].values
    return X, Y


def train_with_model(df_train, model_dict, modelname, n_fold, fold):

    X, Y = create_train_XY(df_train)
    model = model_dict[modelname]
    index = np.arange(len(X))
    match modelname:
        case 'lgb':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
                callbacks=[lgb.early_stopping(100)]
            )
        case 'xgb':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
            )
        case 'cbt':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
                early_stopping_rounds = 100
            )
    # models.append(model)
    joblib.dump(model, f'./models/{modelname}_{fold}.model')
    return model

def train(df_train, n_fold = 5):
    os.system('mkdir models')
    models = []
    for i in range(n_fold):
        print(f"Training fold {i}")
        models.append(train_with_model(df_train, model_dict, 'lgb', n_fold, i))
        # train(model_dict, 'xgb', i)
        # train(model_dict, 'cbt', i)
    
    return models