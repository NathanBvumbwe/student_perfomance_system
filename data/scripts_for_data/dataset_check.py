import pandas as pd

# Load the dataset
df = pd.read_csv('data\\student_performance_dataset_bigger.csv')

# 1. See the first few rows
print(df.head())

# 2. Check data types and non-null counts (Your pd.info)
# For 1 million rows, 'memory_usage="deep"' gives the real RAM impact
print(df.info(memory_usage='deep'))

# 3. Check for any missing values specifically
print(df.isnull().sum())