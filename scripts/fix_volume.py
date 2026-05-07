"""Fix volume unit: akshare returns 手(100shares), old source likely 股(shares).
Multiply post-boundary data by 100 to normalize to shares."""
import pandas as pd
import os, glob, numpy as np

DAILY_K = 'data/local_lake/daily_k'

def detect_and_fix_volume(df, sym):
    vol = df['volume'].astype(float)
    ratio = vol.shift(1) / vol.replace(0, np.nan)
    drops = ratio[ratio > 15].dropna()
    
    if len(drops) == 0:
        return df, None
    
    for bidx in drops.index:
        dt = df.loc[bidx, 'date']
        before = vol.loc[:bidx].iloc[-2] if bidx > 0 else vol.iloc[0]
        after = vol.loc[bidx]
        
        # Check if ×100 fixes the gap
        if 0.5 < after * 100 / before < 2.0 and before / after > 15:
            df.loc[df.index >= bidx, 'volume'] = (df.loc[df.index >= bidx, 'volume'] * 100).astype(int)
            return df, f"×100 from {dt} (gap={before/after:.0f}x, fixed={after*100/before:.1f}x)"
    
    return df, None

# Fix all CSI stocks
constituents = pd.read_parquet('data/local_lake/basics/csi_all_constituents.parquet')
target = set(constituents['symbol'].tolist())

fixed, checked = 0, 0
for f in glob.glob(os.path.join(DAILY_K, '*.parquet')):
    sym = os.path.basename(f).replace('.parquet', '')
    if sym not in target:
        continue
    checked += 1
    df = pd.read_parquet(f)
    df2, note = detect_and_fix_volume(df, sym)
    if note:
        df2.to_parquet(f, index=False)
        fixed += 1
        if fixed <= 8:
            print(f"  [{sym}] {note}")

print(f"\nChecked: {checked}, Fixed: {fixed}")

# Re-verify
if fixed > 0:
    print("\n── RE-VERIFY ──")
    for sym in ['000001', '600519', '601398']:
        df = pd.read_parquet(f'{DAILY_K}/{sym}.parquet')
        april = df[(df['date'] >= '2026-04-01') & (df['date'] <= '2026-04-30')]
        v_before = april[april['date'] < '2026-04-22']['volume'].mean()
        v_after = april[april['date'] >= '2026-04-22']['volume'].mean()
        ratio = v_before / v_after if v_after > 0 else 0
        status = "✅" if ratio < 3 else "❌"
        print(f"  {sym}: avg_vol before={v_before:,.0f}  after={v_after:,.0f}  ratio={ratio:.1f}x {status}")
