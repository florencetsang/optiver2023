def get_df_memory_usage_mb(df):
    return df.memory_usage(index=True).sum() / 1024 / 1024

def get_df_summary_str(df):
    return f"shape: {df.shape}, mem: {get_df_memory_usage_mb(df)} MB"
