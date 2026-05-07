import pandas as pd
import glob

# Benchmark range
bench = pd.read_parquet('data/local_lake/daily_k/sh.000300.parquet')
print(f"Benchmark sh.000300: {len(bench)} rows")
print(f"  Range: {bench['date'].min()} ~ {bench['date'].max()}")
print(f"  Last 5: {list(bench['date'].tail(5))}")

# Sample stock
df = pd.read_parquet('data/local_lake/daily_k/000001.parquet')
print(f"\nStock 000001: {len(df)} rows")
print(f"  Range: {df['date'].min()} ~ {df['date'].max()}")

# Coverage stats for CSI constituents
constituents = pd.read_parquet('data/local_lake/basics/csi_all_constituents.parquet')
csi_set = set(constituents['symbol'].tolist())

daily_k_files = glob.glob('data/local_lake/daily_k/*.parquet')
counts = {'has_202601': 0, 'has_202603': 0, 'total_checked': 0}

for f in daily_k_files:
    sym = f.split('/')[-1].replace('.parquet', '')
    # Match against CSI constituents (strip prefix)
    if sym not in csi_set and sym[3:] not in csi_set:
        continue
    counts['total_checked'] += 1
    d = pd.read_parquet(f)
    max_d = d['date'].max()
    if max_d >= pd.Timestamp('2026-01-01'):
        counts['has_202601'] += 1
    if max_d >= pd.Timestamp('2026-03-31'):
        counts['has_202603'] += 1

print(f"\nCSI constituents with daily_k data: {counts['total_checked']}")
print(f"  Have data through 2026-01-01: {counts['has_202601']} ({100*counts['has_202601']/counts['total_checked']:.0f}%)")
print(f"  Have data through 2026-03-31: {counts['has_202603']} ({100*counts['has_202603']/counts['total_checked']:.0f}%)")
