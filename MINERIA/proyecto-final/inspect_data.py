import pandas as pd

try:
    df = pd.read_csv('test_scores.xls')
    print("First 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nDescription:")
    print(df.describe())
    print("\nColumns:")
    print(df.columns.tolist())
except Exception as e:
    print(f"Error reading file: {e}")
