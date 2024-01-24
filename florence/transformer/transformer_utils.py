import numpy as np

def get_target_col_mask(data_arr, target_col_idx):
    n_cols = data_arr.shape[2]
    target_col_mask = np.zeros(n_cols, bool)
    target_col_mask[target_col_idx] = 1
    return target_col_mask

def get_batch(data_arr, target_col_idx, batch_idx, batch_size):
    # data_arr: [stock-date combination, time per stock-date (i.e. 55), # of features + tgt col]
    # e.g. training set: [95236, 55, 27 + 1]
    #
    # output: (
    #     [batch_size, window_size, # of features] (e.g. [20, 55, 27]),
    #     [batch_size, window_size] (e.g. [20, 55])
    # )
    # e.g. [20, 55, 27]
    #
    # current impl: batch 0 = 0 - 20, batch 1 = 20 - 40, ...
    # TODO: randomized / smarter batching
    
    # TODO: work on window size, full window of 55 now
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, data_arr.shape[0])
    batch = data_arr[start_idx:end_idx]
    
    # construct targets
    target_col_mask = get_target_col_mask(batch, target_col_idx)
    targets = batch[:, :, target_col_mask]
    # flatten targets
    targets = targets.squeeze(-1)

    # construct features
    feature_col_mask = ~target_col_mask
    features = batch[:, :, feature_col_mask]
    features = features.astype("float32")

    return features, targets
