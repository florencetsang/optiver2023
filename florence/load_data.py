import pandas as pd

def load_data_from_csv(data_path):

    df_train = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/train.csv')
    df_test = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/test.csv')
    revealed_targets = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
    sample_submission = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/sample_submission.csv')

    return df_train, df_test, revealed_targets, sample_submission