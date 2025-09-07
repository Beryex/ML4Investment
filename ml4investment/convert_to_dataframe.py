import pickle
import pandas as pd
with open("data/fetched_data.pkl", "rb") as f:
    merged_data = pickle.load(f)

all_dfs = []
for stock_code, df in merged_data.items():
    df_copy = df.copy()
    df_copy['stock_code'] = stock_code
    all_dfs.append(df_copy)

combined_df = pd.concat(all_dfs).sort_index()
combined_df.to_parquet("data/fetched_data.parquet", index=True)