import joblib

class ModelPostProcessor:
    def process(self, model, model_pipeline, fold):
        pass

class CompositeModelPostProcessor(ModelPostProcessor):
    def __init__(self, processors):
        super().__init__()
        self.processors = processors

    def process(self, model, model_pipeline, fold):
        for processor in self.processors:
            processor.process(model, model_pipeline, fold)

class SaveModelPostProcessor(ModelPostProcessor):
    def __init__(self, save_dir='./models/') -> None:
        super().__init__()
        self.save_dir = save_dir
    
    def process(self, model, model_pipeline, fold):
        if model is not None:
            joblib.dump(model, f'{self.save_dir}/{model_pipeline.get_name()}_{fold}.model')
