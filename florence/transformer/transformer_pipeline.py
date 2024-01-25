import math
import time
import torch
from torch import nn

from transformer.transformer_utils import get_batch, get_batch_gpu

from utils.ml_utils import ModelLogger

class TransformerPipeline:
    def __init__(self, model: nn.Module, optimizer, criterion, pipeline_logger: ModelLogger, train_logger: ModelLogger, eval_logger: ModelLogger, test_logger: ModelLogger, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.pipeline_logger = pipeline_logger
        self.train_logger = train_logger
        self.eval_logger = eval_logger
        self.test_logger = test_logger
        self.device = device
        self.pipeline_logger.log(f"device: {device}")

    def _get_total_num_of_batches(self, data_arr, batch_size, first_n_batches_only=-1):
        if first_n_batches_only > -1:
            return first_n_batches_only
        return math.ceil(data_arr.shape[0] / batch_size)

    def train_transformer(self, data_arr, target_col_idx, batch_size, first_n_batches_only=-1):
        self.model.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()
        
        data_arr = data_arr.astype("float32")
        data_tensor = torch.from_numpy(data_arr).to(self.device)

        n_batches = self._get_total_num_of_batches(data_arr, batch_size, first_n_batches_only)
        processed_samples = 0
        for batch_idx in range(n_batches):
            data, targets = get_batch_gpu(data_tensor, target_col_idx, batch_idx, batch_size)
            # apply transformer model
            # output: [batch_size, window_size] (e.g. [20, 55]), matching expected targets
            output = self.model(data, logger=self.train_logger)
            # compute mae
            loss = self.criterion(output, targets)
            # accumulate loss
            loss_val = loss.item()
            n_samples = targets.shape[0]
            processed_samples += n_samples
            total_loss += loss_val * targets.shape[0]
            self.pipeline_logger.log(f"targets: {targets.shape}, output: {output.shape}, loss_val: {loss_val}, total_loss: {total_loss}, n_samples: {n_samples}, processed_samples: {processed_samples}")

            # self.pipeline_logger.log(f"test 1 {self.model.final_linear.bias} {loss.grad} {self.model.final_linear.bias.grad}")
            self.optimizer.zero_grad()
            loss.backward()
            # self.pipeline_logger.log(f"test 2 {self.model.final_linear.bias} {loss.grad} {self.model.final_linear.bias.grad}")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            # self.pipeline_logger.log(f"test 3 {self.model.final_linear.bias} {loss.grad} {self.model.final_linear.bias.grad} {grad_norm}")
            self.optimizer.step()
            # self.pipeline_logger.log(f"test 4 {self.model.final_linear.bias} {loss.grad} {self.model.final_linear.bias.grad} {grad_norm}")

        if processed_samples == 0:
            return 0.0
        return total_loss / processed_samples

    def evaluate_transformer(self, data_arr, target_col_idx, batch_size):
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.0
        
        data_arr = data_arr.astype("float32")
        data_tensor = torch.from_numpy(data_arr).to(self.device)

        with torch.no_grad():
            n_batches = self._get_total_num_of_batches(data_arr, batch_size, -1)
            for batch_idx in range(n_batches):
                data, targets = get_batch_gpu(data_tensor, target_col_idx, batch_idx, batch_size)
                # apply transformer model
                # output: [batch_size, window_size] (e.g. [20, 55]), matching expected targets
                output = self.model(data, logger=self.eval_logger)
                # compute mae
                loss = self.criterion(output, targets)
                
                # accumulate loss
                total_loss += loss.item() * targets.shape[0]

        return total_loss / data_arr.shape[0]

    def test_transformer(self, data_arr):
        self.model.eval()  # turn on evaluation mode
        with torch.no_grad():
            # convert numpy array to pytorch tensor
            data_arr = torch.from_numpy(data_arr).to(self.device)
            # apply transformer model
            # output: [batch_size, window_size] (e.g. [1, 10] for 10th prediction during testing)
            # batch_size must be 1 (predicting one at a time)
            # only the last number is significant, which captures information from sample 1 - 10 features
            output = self.model(data_arr, logger=self.test_logger)
            output = output.cpu()
            output = output.numpy()
            output = output[0, -1]
        return output
