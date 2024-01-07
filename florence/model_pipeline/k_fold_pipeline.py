from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory
import numpy as np
import joblib

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
