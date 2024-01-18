import pandas as pd

def load_data_from_csv(data_path, load_df_train = True, load_df_test = True, load_revealed_targets = True, load_sample_submission = True):

    df_train = None
    df_test = None
    revealed_targets = None
    sample_submission = None

    if load_df_train:
        df_train = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/train.csv')
    if load_df_test: 
        df_test = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/test.csv')
    if load_revealed_targets:
        revealed_targets = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
    if load_sample_submission:
        sample_submission = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/sample_submission.csv')

    return df_train, df_test, revealed_targets, sample_submission