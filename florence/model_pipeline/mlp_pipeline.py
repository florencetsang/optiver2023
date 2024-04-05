from sklearn.neural_network import MLPRegressor
from model_pipeline.model_pipeline import ModelPipeline, ModelPipelineFactory

import torch
from torch import nn, Tensor

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch import loggers as pl_loggers

seed_everything(0, workers=True)

# https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
# https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class OptiverDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        assert x.shape[0] == y.shape[0]
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x.iloc[idx, :], self.y.iloc[idx]

class MLPModel(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes,
        num_input_features: int,
    ):
        super().__init__()
        self.model_type = 'MLP'
        self.layers = []
        prev_layer_size = num_input_features
        for hidden_layer_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(prev_layer_size, hidden_layer_size))
            self.layers.append(nn.ReLU())
            prev_layer_size = hidden_layer_size
        self.layers.append(nn.Linear(prev_layer_size, 1))

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

class MLPModule(L.LightningModule):
    def __init__(self, model, criterion, validation_criterion, lr):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.validation_criterion = validation_criterion
        self.lr = lr
        self.validation_step_outputs = []
        self.validation_step_actual_targets = []

    def training_step(self, batch, batch_idx):
        features, item_id, targets = batch[0], batch[1], batch[2]
        actual_targets = targets[:, -1]
        output = self.model(features, item_id)
        loss = self.criterion(output, actual_targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, item_id, targets = batch[0], batch[1], batch[2]
        actual_targets = targets[:, -1]
        output = self.model(features, item_id)
        # TODO: is loss being averaged based on batch size
        loss = self.validation_criterion(output, actual_targets)
        # lightning will take weighted-average on loss per step based on batch size
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(output)
        self.validation_step_actual_targets.append(actual_targets)

    def on_validation_epoch_end(self):
        # TODO: remove manual calculation of validation loss if we can confirm lightning will take weighted average
        # cat is used instead of stack, last step may have different batch size
        all_preds = torch.cat(self.validation_step_outputs)
        all_actual_targets = torch.cat(self.validation_step_actual_targets)
        manual_loss = self.validation_criterion(all_preds, all_actual_targets)
        self.log("val_loss_manual", manual_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()  # free memory
        self.validation_step_actual_targets.clear()  # free memory

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-1,
        )
        return [optimizer]

class MLPModelPipeline(ModelPipeline):
    # constants
    criterion = nn.L1Loss()
    validation_criterion = nn.L1Loss()

    limit_train_batches = 1.0
    # limit_train_batches = 100
    gradient_clip_val = 0.5

    def __init__(self):
        super().__init__()

    def init_model(self, param: dict = None):
        # self.model = MLPRegressor(**param)

        self.param = param

        model = MLPModel(param["hidden_layer_sizes"], 13)
        model.apply(init_weights)

        self.model = MLPModule(
            model,
            self.criterion,
            self.validation_criterion,
            param["learning_rate_init"],
        )

        model_folder_version_name = None
        # change to a string for a specific name
        # model_folder_version_name = "test"

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html
        # https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        tb_logger = pl_loggers.TensorBoardLogger(".", version=model_folder_version_name)

        self.trainer = L.Trainer(
            max_epochs=self.params["max_iter"],
            limit_train_batches=self.limit_train_batches,
            # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#gradient-clipping
            gradient_clip_val=self.gradient_clip_val,
            callbacks=[
                # https://lightning.ai/docs/pytorch/stable/common/progress_bar.html#richprogressbar
                RichProgressBar(leave=True),
            ],
            logger=tb_logger,
            # https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
            deterministic=True,
        )

    def train(self, train_X, train_Y, eval_X, eval_Y, eval_res):
        # self.model.fit(train_X, train_Y)

        full_training_dataset = OptiverDataset(train_X, train_Y)
        full_validation_dataset = OptiverDataset(eval_X, eval_Y)
        training_sampler = torch.utils.data.RandomSampler(full_training_dataset)
        validation_sampler = torch.utils.data.SequentialSampler(full_validation_dataset)
        training_dataloader = torch.utils.data.DataLoader(
            full_training_dataset,
            batch_size=self.param["batch_size"],
            sampler=training_sampler,
            # https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
            num_workers=4,
            # https://pytorch.org/docs/stable/data.html#memory-pinning
            pin_memory=True,
        )
        validation_dataloader = torch.utils.data.DataLoader(
            full_validation_dataset,
            batch_size=self.param["batch_size"],
            sampler=validation_sampler,
            num_workers=4,
            pin_memory=True,
        )
        self.trainer.fit(
            model=self.model,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )
    
    def get_name(self):
        return "mlp"


    def get_name_with_params(self, params):
        selected_params_for_model_name = ['learning_rate', 'max_depth', 'n_estimators']
        return "_".join([f"{param_n}_{params[param_n]}" for param_n in selected_params_for_model_name])

    def get_hyper_params(self, trial):
        return {
            'hidden_layer_sizes' : [128, 256, 128],
            # 'activation' : 'relu',
            # 'solver' : 'adam',
            # 'alpha' : 0.0,
            'batch_size' : 10,
            # 'random_state' : 0,
            # 'tol' : 0.0001,
            # 'nesterovs_momentum' : False,
            # 'learning_rate' : 'constant',
            'learning_rate_init' : trial.suggest_float('learning_rate_init', 0.001, 0.01, log=True),
            'max_iter' : 200,
            # 'shuffle' : True,
            'n_iter_no_change' : 10,
            'early_stopping': True,
            # 'verbose' : False 
        }

class MLPModelPipelineFactory(ModelPipelineFactory):
    def create_model_pipeline(self) -> ModelPipeline:
        return MLPModelPipeline()
