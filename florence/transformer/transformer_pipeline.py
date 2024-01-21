import time
import torch
from torch import nn

from transformer.transformer_utils import get_batch

from utils.ml_utils import ModelLogger

class TransformerPipeline:
    def __init__(self, model: nn.Module, optimizer, criterion, train_logger: ModelLogger, eval_logger: ModelLogger, test_logger: ModelLogger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_logger = train_logger
        self.eval_logger = eval_logger
        self.test_logger = test_logger

    def train_transformer(self, data_arr, target_col_idx, n_batches, batch_size):
        self.model.train()  # turn on train mode
        total_loss = 0.0
        start_time = time.time()

        for batch_idx in range(n_batches):
            data, targets = get_batch(data_arr, target_col_idx, batch_idx, batch_size)
            # convert numpy array to pytorch tensor
            data = torch.from_numpy(data)
            targets = torch.from_numpy(targets)
            # apply transformer model
            # output: [batch_size, window_size] (e.g. [20, 55]), matching expected targets
            output = self.model(data, logger=self.train_logger)
            self.train_logger.log(f"targets: {targets.shape}, output: {output.shape}")
            # compute mae
            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            # accumulate loss
            total_loss += loss.item()
        
        return total_loss

    def evaluate_transformer(self, data_arr, target_col_idx, batch_size):
        self.model.eval()  # turn on evaluation mode
        total_loss = 0.0
        eval_batches_cnt = 0

        with torch.no_grad():
            for batch_idx in range(0, data_arr.shape[0], batch_size):
                eval_batches_cnt += 1

                data, targets = get_batch(data_arr, target_col_idx, batch_idx, batch_size)
                # convert numpy array to pytorch tensor
                data = torch.from_numpy(data)
                targets = torch.from_numpy(targets)
                # apply transformer model
                # output: [batch_size, window_size] (e.g. [20, 55]), matching expected targets
                output = self.model(data, logger=self.eval_logger)
                # compute mae
                loss = self.criterion(output, targets)
                
                # accumulate loss
                total_loss += loss.item()

        if eval_batches_cnt <= 0:
            return 0.0
        return total_loss / eval_batches_cnt

    def test_transformer(self, data_arr):
        self.model.eval()  # turn on evaluation mode
        with torch.no_grad():
            # convert numpy array to pytorch tensor
            data_arr = torch.from_numpy(data_arr)
            # apply transformer model
            # output: [batch_size, window_size] (e.g. [1, 10] for 10th prediction during testing)
            # batch_size must be 1 (predicting one at a time)
            # only the last number is significant, which captures information from sample 1 - 10 features
            output = self.model(data_arr, logger=self.test_logger)
            output = output.numpy()
            output = output[0, -1]
        return output
