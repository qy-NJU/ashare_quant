import os
import sys
import time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.source.tushare_source import TushareSource

PARQUET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_lake', 'daily_k')

# Load CSI constituents
constituents = pd.read_parquet(os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'local_lake', 'basics', 'csi_all_constituents.parquet'
))
all_symbols = constituents['symbol'].tolist()

# Add benchmark
all_symbols.append('sh.000300')

# Deduplicate
all_symbols = sorted(set(all_symbols))
print(f"Target: {len(all_symbols)} symbols (CSI 300+500+1000 + benchmark)")

# Check which need updating
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
to_update = []
up_to_date = 0
no_file = 0

for sym in all_symbols:
    path = os.path.join(PARQUET_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        # Also try without prefix
        alt_sym = sym.replace('sh.', '').replace('sz.', '').replace('bj.', '')
        alt_path = os.path.join(PARQUET_DIR, f"{alt_sym}.parquet")
        if os.path.exists(alt_path):
            path = alt_path
        else:
            no_file += 1
            to_update.append((sym, '2015-01-01'))
            continue

    df = pd.read_parquet(path)
    if 'date' in df.columns:
        last_date = pd.to_datetime(df['date']).max()
    else:
        last_date = df.index.max()

    last_str = last_date.strftime('%Y-%m-%d')
    if last_str >= end_date:
        up_to_date += 1
    else:
        next_day = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        to_update.append((sym, next_day))

print(f"Up to date: {up_to_date}")
print(f"Need update: {len(to_update)}")
print(f"No file: {no_file}")

if not to_update:
    print("All data is up to date!")
    sys.exit(0)

# Start sync
source = TushareSource()
print(f"\nStarting incremental sync for {len(to_update)} symbols...")

success = 0
failed = 0
batch_size = 50

for i in range(0, len(to_update), batch_size):
    batch = to_update[i:i+batch_size]
    batch_syms = [s for s, d in batch]
    start_date = batch[0][1]  # Use first symbol's date as batch start

    try:
        batch_data = source.get_daily_data_batch(batch_syms, start_date=start_date, end_date=end_date)

        for sym, _start in batch:
            save_path = os.path.join(PARQUET_DIR, f"{sym}.parquet")
            # Try alternate path
            if not os.path.exists(save_path):
                alt = sym.replace('sh.', '').replace('sz.', '').replace('bj.', '')
                alt_path = os.path.join(PARQUET_DIR, f"{alt}.parquet")
                if os.path.exists(alt_path):
                    save_path = alt_path

            new_df = batch_data.get(sym, pd.DataFrame())

            if not new_df.empty:
                local_df = pd.DataFrame()
                if os.path.exists(save_path):
                    local_df = pd.read_parquet(save_path)

                if 'date' in local_df.columns and not local_df.empty:
                    local_df['date'] = pd.to_datetime(local_df['date'])
                if 'date' in new_df.columns and not new_df.empty:
                    new_df['date'] = pd.to_datetime(new_df['date'])

                if not local_df.empty:
                    combined = pd.concat([local_df, new_df]).drop_duplicates('date', keep='last').sort_values('date')
                else:
                    combined = new_df

                combined.to_parquet(save_path, index=False)
                success += 1
            else:
                # No new data, still count as success
                success += 1

    except Exception as e:
        for sym, _start in batch:
            failed += 1
        if failed <= 5:
            print(f"  Batch failed: {e}")

    time.sleep(0.5)  # Rate limit
    print(f"  Progress: {min(i+batch_size, len(to_update))}/{len(to_update)} (success={success}, failed={failed})")

print(f"\nCompleted: success={success}, failed={failed}")
