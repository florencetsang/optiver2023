import numpy as np

class DataPreprocessor:
    def apply(self, df):
        return df
    
class CompositeDataPreprocessor(DataPreprocessor):
    def __init__(self, processors):
        self.processors = processors

    def apply(self, df):
        processed_df = df
        for processor in self.processors:
            processed_df = processor.apply(processed_df)
        return processed_df

# TODO: Is it a preprocessor?
class TimeSeriesDataPreprocessor:
    def apply(self, df):
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
        df_sorted = df.sort_values("index_col_id", ignore_index=True)

        # https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
        # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        index_col_id_uniq, index_col_id_uniq_idx = np.unique(df_sorted["index_col_id"].values, axis=0, return_index=True)
        df_train_arr = np.split(df_sorted.values, index_col_id_uniq_idx[1:])
        # https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        df_train_arr = np.stack(df_train_arr, axis=0)
        # df_train_arr: [stock-date combination, time per stock-date (i.e. 55), # of features]

        return df_train_arr

class ReduceMemUsageDataPreprocessor(DataPreprocessor):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def apply(self, df):
        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)

        if self.verbose:
            logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
            end_mem = df.memory_usage().sum() / 1024**2
            logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
            decrease = 100 * (start_mem - end_mem) / start_mem
            logger.info(f"Decreased by {decrease:.2f}%")

        return df