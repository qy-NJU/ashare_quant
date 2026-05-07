"""Deep dive: cross-source volume consistency at the overlap boundary (April 2026)."""
import pandas as pd

STOCKS = ['000001', '600519', '601398', '300750', '688981', '601318']
DAILY_K = 'data/local_lake/daily_k'

print("=" * 70)
print("VOLUME CROSS-SOURCE CONSISTENCY CHECK")
print("Comparing April 2026 (overlap zone between old & new source)")
print("=" * 70)

for sym in STOCKS:
    df = pd.read_parquet(f'{DAILY_K}/{sym}.parquet')
    april = df[(df['date'] >= '2026-04-01') & (df['date'] <= '2026-04-30')].copy()
    
    if april.empty:
        continue

    # Calculate daily volume change
    april['vol_chg'] = april['volume'].pct_change().abs()
    
    # Mark the likely cutoff point (where vol drops dramatically)
    vol_mean_before_22 = april[april['date'] < '2026-04-22']['volume'].mean()
    vol_mean_after_22 = april[april['date'] >= '2026-04-22']['volume'].mean()
    
    ratio = vol_mean_before_22 / vol_mean_after_22 if vol_mean_after_22 > 0 else float('inf')
    
    print(f"\n── {sym} ──")
    print(f"  avg vol before 4/22: {vol_mean_before_22:,.0f}")
    print(f"  avg vol after  4/22:  {vol_mean_after_22:,.0f}")
    print(f"  ratio (before/after): {ratio:.1f}x")
    
    if ratio > 10:
        print(f"  ❌ VOLUME DISCONTINUITY at 4/22 boundary! (>10x drop)")
    elif ratio > 2:
        print(f"  ⚠️  Volume changed >2x at 4/22 boundary")
    else:
        print(f"  ✅ Volume consistent across source boundary")
    
    print(f"  Daily:")
    for _, row in april.iterrows():
        dt = row['date'].strftime('%Y-%m-%d')
        marker = " ← NEW SOURCE?" if dt >= '2026-04-22' else " [OLD]"
        print(f"    {dt}  close={row['close']:8.2f}  vol={row['volume']:>14,.0f}{marker}")

# Also check: does this vol drop correspond to actual market low volume?
# Check HS300 volume in April
print("\n── HS300 BENCHMARK APRIL ──")
b = pd.read_parquet(f'{DAILY_K}/sh.000300.parquet')
b_april = b[(b['date'] >= '2026-04-01') & (b['date'] <= '2026-04-30')]
for _, row in b_april.iterrows():
    dt = row['date'].strftime('%Y-%m-%d')
    marker = " ← NEW?" if dt >= '2026-04-22' else ""
    print(f"  {dt}  close={row['close']:8.0f}  vol={row['volume']:>14,.0f}{marker}")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("  If ALL stocks show same volume pattern, it's a market-wide phenomenon")
print("  If only individual stocks show it, it's a source unit mismatch")
