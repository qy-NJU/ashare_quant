"""Analyze low win-rate root causes for CatBoost 2026 backtest."""
import pandas as pd
import numpy as np

TRADES = 'data/backtest/trades_20260101_20260430.csv'
BENCH = 'data/local_lake/daily_k/sh.000300.parquet'

# ── 1. Strategy Config ──
print("=" * 70)
print("1. STRATEGY CONFIG")
print("=" * 70)
import yaml
with open('configs/pipeline_config.yaml') as f:
    cfg = yaml.safe_load(f)['pipeline']
print(f"  Model: {cfg['model']['name']}")
print(f"  TopK: {cfg['strategy']['params'].get('top_k')}")
print(f"  Rebalance: {cfg['strategy']['params'].get('rebalance_period')}d")
print(f"  Weight: {cfg['strategy']['params'].get('weight_method')}")
print(f"  Target position: {cfg['strategy']['params'].get('target_position_ratio')}")
print(f"  ATR stop mult: {cfg['strategy']['params'].get('atr_stop_mult')}")
print(f"  Take-profit activate: {cfg['strategy']['params'].get('take_profit_activate')}")

# ── 2. Market Environment ──
print("\n" + "=" * 70)
print("2. MARKET ENVIRONMENT (HS300)")
print("=" * 70)
bench = pd.read_parquet(BENCH)
b2026 = bench[(bench['date'] >= '2026-01-01') & (bench['date'] <= '2026-04-30')].copy()
s, e = b2026.iloc[0]['close'], b2026.iloc[-1]['close']
mx, mn = b2026['close'].max(), b2026['close'].min()
print(f"  HS300: {s:.0f} -> {e:.0f} ({(e/s-1)*100:+.1f}%)")
print(f"  Peak: {mx:.0f}, Trough: {mn:.0f}, MaxDD: {(mn/mx-1)*100:+.1f}%")

b2026['month'] = b2026['date'].dt.strftime('%Y-%m')
for m in sorted(b2026['month'].unique()):
    md = b2026[b2026['month'] == m]
    ret = md.iloc[-1]['close'] / md.iloc[0]['close'] - 1
    print(f"  {m}: {md.iloc[0]['close']:.0f}->{md.iloc[-1]['close']:.0f} ({ret:+.1f}%)")

up = (b2026['close'].diff() > 0).sum()
dn = (b2026['close'].diff() < 0).sum()
print(f"  Up days: {up}, Down days: {dn}, U/D ratio: {up/dn:.2f}")

# ── 3. Trade-by-trade PnL Analysis ──
print("\n" + "=" * 70)
print("3. TRADE-BY-TRADE PnL")
print("=" * 70)
trades = pd.read_csv(TRADES)

# Reconstruct round-trips
winners, losers = [], []
for sym in sorted(trades['symbol'].unique()):
    st = trades[trades['symbol'] == sym].sort_values('date')
    b = st[st['action'] == 'BUY']
    s = st[st['action'] == 'SELL']
    if b.empty or s.empty:
        continue
    total_buy_amt = (b['qty'] * b['price']).sum()
    total_buy_qty = b['qty'].sum()
    avg_buy = total_buy_amt / total_buy_qty
    total_sell_amt = (s['qty'] * s['price']).sum()
    total_sell_qty = s['qty'].sum()
    avg_sell = total_sell_amt / total_sell_qty
    pnl_pct = (avg_sell - avg_buy) / avg_buy * 100
    entry_date = b['date'].min()
    exit_date = s['date'].max()
    holding = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
    entry = (entry_date, avg_buy, avg_sell, pnl_pct, holding, sym)
    if pnl_pct > 0:
        winners.append(entry)
    else:
        losers.append(entry)

print(f"\n  Winners: {len(winners)}/{len(winners)+len(losers)} ({100*len(winners)/(len(winners)+len(losers)):.0f}%)")
print(f"  Avg win: +{np.mean([w[3] for w in winners]):.1f}%, Avg loss: {np.mean([l[3] for l in losers]):.1f}%")
print(f"  Avg holding (win): {np.mean([w[4] for w in winners]):.0f}d, (loss): {np.mean([l[4] for l in losers]):.0f}d")

print("\n  🟢 Winners:")
for w in sorted(winners, key=lambda x: x[3], reverse=True):
    print(f"    {w[5]:<8} in={w[0]} hold={w[4]:>2}d PnL=+{w[3]:.1f}% buy={w[1]:.2f} sell={w[2]:.2f}")

print("\n  🔴 Losers:")
for l in sorted(losers, key=lambda x: x[3]):
    print(f"    {l[5]:<8} in={l[0]} hold={l[4]:>2}d PnL={l[3]:.1f}% buy={l[1]:.2f} sell={l[2]:.2f}")

# ── 4. Monthly breakdown ──
print("\n" + "=" * 70)
print("4. MONTHLY PNL BREAKDOWN")
print("=" * 70)
all_trades_data = winners + losers
all_trades_data.sort(key=lambda x: x[0])
monthly = {}
for e in all_trades_data:
    m = str(e[0])[:6]
    monthly.setdefault(m, {'wins': [], 'losses': []})
    if e[3] > 0:
        monthly[m]['wins'].append(e[3])
    else:
        monthly[m]['losses'].append(e[3])

for m in sorted(monthly):
    d = monthly[m]
    print(f"  {m}: {len(d['wins'])}W/{len(d['losses'])}L  avg_win=+{np.mean(d['wins']) if d['wins'] else 0:.1f}%  avg_loss={np.mean(d['losses']) if d['losses'] else 0:.1f}%")

# ── 5. Stop-loss analysis ──
print("\n" + "=" * 70)
print("5. STOP-LOSS / EXIT REASON ANALYSIS")
print("=" * 70)
# Check sell reasons from trade log
sell_reasons = trades[trades['action']=='SELL']['reason'].value_counts()
print(f"  Sell reasons: {dict(sell_reasons)}")

# ── 6. Market beta check: how did market do when we entered?
print("\n" + "=" * 70)
print("6. ENTRY TIMING vs MARKET")
print("=" * 70)
for e in winners[:3] + losers[:3]:
    sym, entry_d, pnl = e[5], e[0], e[3]
    # What was HS300 doing in the next 5 days after entry?
    try:
        idx = b2026[b2026['date'] == entry_d].index
        if len(idx) > 0:
            pos = b2026.index.get_loc(idx[0])
            fut_5d = b2026.iloc[min(pos+5, len(b2026)-1)]['close'] / b2026.iloc[pos]['close'] - 1
            tag = "🟢" if pnl > 0 else "🔴"
            print(f"  {tag} {sym} entered {entry_d}: PnL={pnl:+.1f}%, HS300 next 5d={fut_5d*100:+.1f}%")
    except:
        pass
