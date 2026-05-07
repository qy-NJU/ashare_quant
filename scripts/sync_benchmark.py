import akshare as ak
import pandas as pd

df = ak.stock_zh_index_daily(symbol='sh000300')
df['date'] = pd.to_datetime(df['date'])
df_clean = df[['date','open','close','high','low','volume']]

old = pd.read_parquet('data/local_lake/daily_k/sh.000300.parquet')
print(f"old: {len(old)} rows, last={old['date'].max()}")

combined = pd.concat([old, df_clean]).drop_duplicates('date', keep='last').sort_values('date')
combined.to_parquet('data/local_lake/daily_k/sh.000300.parquet', index=False)
print(f"new: {len(combined)} rows, last={combined['date'].max()}")
