import pandas as pd
import numpy as np
import plotly.express as px
from load_data import load_data_from_csv

# DATA_PATH = '/kaggle/input'
DATA_PATH = '..'
print("loading data")
df_train, _, _, _ = load_data_from_csv(DATA_PATH, load_df_test=False, load_revealed_targets=False, load_sample_submission=False)
print("loaded data")

# 1. Compute returns for each stock for each day
# Calculate daily returns
# def calculate_daily_returns(stock_data):
#     stock_data['return'] = stock_data['wap'].pct_change()
#     return stock_data[['return', 'seconds_in_bucket']].dropna()  # Keep 'return' and 'seconds_in_bucket' columns

# returns2 = df_train.groupby(['stock_id', 'date_id']).apply(calculate_daily_returns).reset_index()

df_train['return'] = df_train.groupby(['stock_id', 'date_id'])['wap'].pct_change()

df_train2 = df_train.dropna(subset=['return', 'seconds_in_bucket'])

# 2. Align the data for each stock by filling in the gaps (if any) and then concatenate the returns to form a matrix
# For this step, we will pivot the data so each stock has its own column, and each row represents a timestamp.
pivot_returns = df_train2.pivot_table(index=['date_id', 'seconds_in_bucket'], 
                                    columns='stock_id', 
                                    values='return')

# handle missing values by filling the average of all available 
pivot_returns = pivot_returns.apply(lambda row: row.fillna(row.mean()), axis=1)

# 3. Compute the correlation matrix for all stocks
correlation_matrix = pivot_returns.corr()

print(correlation_matrix)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Hierarchical clustering
Z = linkage(correlation_matrix, 'ward')
clusters = fcluster(Z, 10, criterion = 'maxclust')

# Assign clusters to stocks
stock_clusters = pd.DataFrame({'stock_id': correlation_matrix.index, 'cluster': clusters})
print(stock_clusters)

plt.figure(figsize=(20, 10))
dendrogram(Z, labels=correlation_matrix.index, leaf_rotation=90)
plt.title('Dendrogram of Stock Clustering based on Correlation')
plt.xlabel('Stocks')
plt.ylabel('Euclidean distances')
plt.axhline(y=1500, color='r', linestyle='--') # This line represents the cut-off for the clusters. Adjust the value as needed.
plt.show()
