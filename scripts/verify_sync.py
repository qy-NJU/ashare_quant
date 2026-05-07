"""Verify data integrity after sync: check volume/price ranges, date gaps, anomalies."""
import pandas as pd
import numpy as np

STOCKS = ['000001', '688981', '600519', '601398', '300750']
DAILY_K = 'data/local_lake/daily_k'
BASICS = 'data/local_lake/basics'

print("=" * 70)
print("DATA SYNC INTEGRITY VERIFICATION")
print("=" * 70)

for sym in STOCKS:
    df = pd.read_parquet(f'{DAILY_K}/{sym}.parquet')
    print(f"\n── {sym} ──")
    print(f"  rows={len(df)}, dates={df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  close: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    print(f"  volume: {df['volume'].min():,.0f} ~ {df['volume'].max():,.0f}")

    # Check date gaps > 5 trading days
    dates = pd.to_datetime(df['date']).sort_values()
    gaps = dates.diff().dt.days
    big_gaps = gaps[gaps > 5]
    if len(big_gaps) > 0:
        for idx in big_gaps.index[:3]:
            print(f"  gap: {dates[idx-1].strftime('%Y-%m-%d')} → {dates[idx].strftime('%Y-%m-%d')} ({big_gaps[idx]:.0f}d)")

    # Volume sanity: pct change > 5x in adjacent days
    vol = df['volume'].astype(float)
    vol_chg = vol.pct_change().abs()
    extreme = vol_chg[vol_chg > 5]
    if len(extreme) > 0:
        print(f"  ⚠️ volume spikes >500%: {len(extreme)} occurrences")

    # Check for zero-volume days (suspension is ok, >10 is suspicious)
    zero_vol = (df['volume'] == 0).sum()
    if zero_vol > 10:
        print(f"  ⚠️ zero-volume days: {zero_vol}")

# ── benchmark sanity ──
print("\n── benchmark sh.000300 ──")
b = pd.read_parquet(f'{DAILY_K}/sh.000300.parquet')
print(f"  rows={len(b)}, dates={b['date'].min().strftime('%Y-%m-%d')} → {b['date'].max().strftime('%Y-%m-%d')}")
print(f"  close: {b['close'].min():.0f} → {b['close'].max():.0f}")
if 2000 < b['close'].min() < b['close'].max() < 7000:
    print(f"  ✅ HS300 price range normal (2000-7000)")
else:
    print(f"  ❌ HS300 price range abnormal!")

# ── Cross-source overlap check ──
print("\n── CROSS-SOURCE OVERLAP CHECK ──")
# Compare 000001 around 2026-04-10 (likely overlap point between old & new source)
df1 = pd.read_parquet(f'{DAILY_K}/000001.parquet')
april = df1[(df1['date'] >= '2026-04-01') & (df1['date'] <= '2026-04-30')]
print(f"\n  000001 April 2026 daily data:")
for _, row in april.iterrows():
    print(f"    {row['date'].strftime('%Y-%m-%d')}  close={row['close']:7.2f}  vol={row['volume']:>12,.0f}  open={row['open']:7.2f}  high={row['high']:7.2f}  low={row['low']:7.2f}")

# ── Price consistency check ──
print("\n── PRICE CONSISTENCY ──")
for sym in STOCKS[:2]:
    df = pd.read_parquet(f'{DAILY_K}/{sym}.parquet')
    # Check no negative prices
    assert (df['close'] > 0).all(), f"{sym}: negative close!"
    assert (df['open'] > 0).all(), f"{sym}: negative open!"
    # Check open/close ratio sanity (no stock moves >50% overnight)
    overnight = (df['open'] / df['close'].shift(1) - 1).abs()
    crazy = overnight[overnight > 0.5].dropna()
    if len(crazy) > 0:
        print(f"  {sym}: {len(crazy)} days with overnight gap >50%")
    else:
        print(f"  {sym}: ✅ no crazy overnight gaps")

# ── Constituent coverage ──
print("\n── CONSTITUENT COVERAGE ──")
import os, glob
files = set(f.replace('.parquet', '') for f in os.listdir(DAILY_K) if f.endswith('.parquet'))
const = pd.read_parquet(f'{BASICS}/csi_all_constituents.parquet')
csi_set = set(const['symbol'].tolist())
in_pool = csi_set & files
print(f"  CSI constituents: {len(csi_set)}")
print(f"  Has daily_k file: {len(in_pool)} ({100*len(in_pool)/len(csi_set):.0f}%)")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
