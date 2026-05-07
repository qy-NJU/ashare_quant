import akshare as ak
import pandas as pd
import os

basics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_lake', 'basics')
os.makedirs(basics_dir, exist_ok=True)

INDEX_MAP = {
    'csi300': '000300',
    'csi500': '000905',
    'csi1000': '000852',
}

for idx_name, idx_code in INDEX_MAP.items():
    print(f"Fetching {idx_name} ({idx_code})...")
    df = ak.index_stock_cons_csindex(symbol=idx_code)

    codes = df['成分券代码'].astype(str).str.zfill(6).tolist()
    result_df = pd.DataFrame({'symbol': sorted(codes)})
    path = os.path.join(basics_dir, f'{idx_name}_constituents.parquet')
    result_df.to_parquet(path, engine='pyarrow')
    print(f'  Saved {len(result_df)} stocks -> {os.path.basename(path)}')

# Combined unique set
all_symbols = set()
for idx_name in INDEX_MAP:
    path = os.path.join(basics_dir, f'{idx_name}_constituents.parquet')
    df = pd.read_parquet(path)
    all_symbols.update(df['symbol'].tolist())

combined = pd.DataFrame({'symbol': sorted(all_symbols)})
combined.to_parquet(os.path.join(basics_dir, 'csi_all_constituents.parquet'), engine='pyarrow')
print(f'\nCombined CSI 300+500+1000 unique: {len(combined)} stocks')
