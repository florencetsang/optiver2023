from data_generator.data_generator import TrainEvalDataGenerator
from model_pipeline.model_pipeline import ModelPipelineFactory
from model_post_processor.model_post_processor import ModelPostProcessor

class DefaultTrainPipeline():
    def __init__(
            self,
            model_pipeline_factory: ModelPipelineFactory,
            train_eval_data_generator: TrainEvalDataGenerator,
            model_post_processor: ModelPostProcessor,
            callbacks
    ):
        self.model_pipeline_factory = model_pipeline_factory
        self.train_eval_data_generator = train_eval_data_generator
        self.model_post_processor = model_post_processor
        self.callbacks = callbacks
    
    def train(self, df_train):
        print(f"generate data")
        train_dfs, eval_dfs, num_train_eval_sets = self.train_eval_data_generator.generate(df_train)

        models = []
        model_res = []

        print(f"start training, num_train_eval_sets: {num_train_eval_sets}")
        for fold in range(num_train_eval_sets):
            print(f"Training fold {fold} - start")

            model_pipeline = self.model_pipeline_factory.create_model_pipeline()
            model_pipeline.init_model()
            print(f"Training fold {fold} - initialized")

            fold_df_train = train_dfs[fold]
            fold_df_eval = eval_dfs[fold]
            print(f"Training fold {fold} - train size: {fold_df_train.shape}, eval size: {fold_df_eval.shape}")

            print(f"Training fold {fold} - start training")
            train_res = model_pipeline.train(fold_df_train, fold_df_eval)
            fold_model = model_pipeline.get_model()
            models.append(fold_model)
            model_res.append(train_res)
            print(f"Training fold {fold} - finished training")
            
            self.model_post_processor.process(fold_model, model_pipeline, fold)
            print(f"Training fold {fold} - finished post processing")

            print(f"Training fold {fold} - end")
        
        print(f"finished training, num_train_eval_sets: {num_train_eval_sets}")

        callback_results = []
        for callback in self.callbacks:
            callback_res = callback.on_callback(models, model_res, train_dfs, eval_dfs, num_train_eval_sets)
            callback_results.append(callback_res)

        return models, model_res, train_dfs, eval_dfs, num_train_eval_sets, callback_results
