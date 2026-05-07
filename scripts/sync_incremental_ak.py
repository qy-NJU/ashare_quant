"""
Incremental sync CSI constituent daily data using akshare.
Only fetches missing days for each stock, skips up-to-date ones.
"""
import os
import time
import pandas as pd
import akshare as ak

DAILY_K_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_lake', 'daily_k')
BASICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_lake', 'basics')

# Load CSI constituent list
df_const = pd.read_parquet(os.path.join(BASICS_DIR, 'csi_all_constituents.parquet'))
target_symbols = sorted(df_const['symbol'].tolist())
target_symbols.append('000300')  # benchmark
print(f"Total targets: {len(target_symbols)}")

# Find which need updating
end_date = pd.Timestamp.now().strftime('%Y%m%d')
to_update = []
up_to_date = 0

for sym in target_symbols:
    path = os.path.join(DAILY_K_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        to_update.append((sym, '20240101'))
        continue
    df = pd.read_parquet(path)
    if 'date' in df.columns:
        last = pd.to_datetime(df['date'].max())
    else:
        last = df.index.max()
    if last.strftime('%Y%m%d') >= end_date:
        up_to_date += 1
    else:
        to_update.append((sym, (last + pd.Timedelta(days=1)).strftime('%Y%m%d')))

print(f"Up to date: {up_to_date}")
print(f"Need update: {len(to_update)}")

if not to_update:
    print("All data is up to date!")
    exit()

# Sync in batches
success = 0
fail = 0
batch_delay = 2  # seconds between batches

for i, (sym, start_d) in enumerate(to_update):
    try:
        # akshare stock_zh_a_hist: symbol is like '000001', period='daily'
        # Returns date, open, close, high, low, volume, amount...
        df_new = ak.stock_zh_a_hist(symbol=sym, period='daily',
                                     start_date=start_d, end_date=end_date,
                                     adjust='qfq')
        if df_new.empty:
            success += 1
            continue

        path = os.path.join(DAILY_K_DIR, f"{sym}.parquet")
        if os.path.exists(path):
            df_old = pd.read_parquet(path)
            df_old['date'] = pd.to_datetime(df_old['date'])
        else:
            df_old = pd.DataFrame()

        df_new['date'] = pd.to_datetime(df_new['日期'])
        df_new = df_new[['date', '开盘', '最高', '最低', '收盘', '成交量']]
        df_new.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        if not df_old.empty:
            combined = pd.concat([df_old, df_new]).drop_duplicates('date', keep='last').sort_values('date')
        else:
            combined = df_new

        combined.to_parquet(path, index=False)
        success += 1

    except Exception as e:
        fail += 1
        if fail <= 3:
            print(f"  {sym} failed: {e}")

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(to_update)} (ok={success}, fail={fail})")
        time.sleep(batch_delay)

print(f"\nDone. ok={success}, fail={fail}")
