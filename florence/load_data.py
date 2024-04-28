import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_from_csv(data_path, load_df_test=True, load_revealed_targets=True, load_sample_submission=True):
    df_train = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/train.csv')
    df_train, df_val = train_test_split(df_train, test_size=0.1, shuffle=False)
    
    if load_df_test:
        df_test = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/test.csv')
        revealed_targets = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/revealed_targets.csv')
        sample_submission = pd.read_csv(f'{data_path}/optiver-trading-at-the-close/example_test_files/sample_submission.csv')
    else:
        df_test, revealed_targets, sample_submission = None, None, None
    
    return df_train, df_val, df_test, revealed_targets, sample_submission