"""
Deep analysis: what happened AFTER stop-loss / forced rebalance?
All three questions answered with real data.
"""
import pandas as pd
import numpy as np
import os

TRADES = 'data/backtest/trades_20260101_20260430.csv'
DAILY_K = 'data/local_lake/daily_k'

# ── Load trades ──
trades = pd.read_csv(TRADES)
trades['date'] = pd.to_datetime(trades['date'].astype(str), format='%Y%m%d')

# ── Build round-trip lifecycles ──
lifecycles = []
pos_qty = {}    # sym -> cumulative qty
pos_cost = {}   # sym -> cumulative cost
pos_entry = {}  # sym -> first entry date

for _, t in trades.sort_values('date').iterrows():
    sym = t['symbol']
    if t['action'] == 'BUY':
        if sym not in pos_qty:
            pos_qty[sym] = 0; pos_cost[sym] = 0.0; pos_entry[sym] = t['date']
        pos_qty[sym] += t['qty']
        pos_cost[sym] += t['qty'] * t['price']
    elif t['action'] == 'SELL':
        if sym not in pos_qty or pos_qty[sym] <= 0:
            continue
        sell_qty = min(t['qty'], pos_qty[sym])
        avg_cost = pos_cost[sym] / pos_qty[sym]
        lifecycles.append({
            'symbol': sym,
            'entry_date': pos_entry[sym],
            'exit_date': t['date'],
            'entry_price': avg_cost,
            'exit_price': t['price'],
            'pnl_pct': (t['price']/avg_cost - 1) * 100,
            'reason': t['reason']
        })
        pos_qty[sym] -= sell_qty
        pos_cost[sym] -= sell_qty * avg_cost
        if pos_qty[sym] <= 0:
            del pos_qty[sym], pos_cost[sym], pos_entry[sym]

df = pd.DataFrame(lifecycles)
print(f"Round-trips: {len(df)}, wins: {(df['pnl_pct']>0).sum()}/{len(df)} ({(df['pnl_pct']>0).mean()*100:.0f}%)")
print(f"Avg win: +{df[df['pnl_pct']>0]['pnl_pct'].mean():.1f}%, Avg loss: {df[df['pnl_pct']<0]['pnl_pct'].mean():.1f}%")

# ═══════════════════════════════════════════
# ANALYSIS 1: What happened AFTER each exit?
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("1. WHAT HAPPENED AFTER STOP-LOSS / SELL?")
print("="*70)

aftermath = []
for _, row in df.iterrows():
    sym, exit_dt, exit_px = row['symbol'], row['exit_date'], row['exit_price']
    f = os.path.join(DAILY_K, f"{sym}.parquet")
    if not os.path.exists(f):
        continue
    stock = pd.read_parquet(f)
    stock['date'] = pd.to_datetime(stock['date'])
    m = stock[stock['date'] == exit_dt]
    if m.empty:
        continue
    p = m.index[0]  # integer row position

    r = {'symbol': sym, 'exit_date': exit_dt.strftime('%Y%m%d'),
         'pnl': row['pnl_pct']}
    for n, label in [(5,'5d'),(10,'10d'),(20,'20d')]:
        fp = min(p+n, len(stock)-1)
        r[f'{label}_ret'] = float((stock.iloc[fp]['close']/exit_px - 1)*100)
        r[f'{label}_date'] = stock.iloc[fp]['date'].strftime('%Y%m%d')
    w = min(p+10, len(stock)-1)
    r['max_10d'] = float((stock.iloc[p:w+1]['close'].max()/exit_px - 1)*100)
    aftermath.append(r)

adf = pd.DataFrame(aftermath)
losses = adf[adf['pnl'] < 0]
wins_exit = adf[adf['pnl'] > 0]

print(f"\n  Loss exits (PnL<0): {len(losses)}")
print(f"  {'Metric':<20} {'5d':>8} {'10d':>8} {'20d':>8}")
print(f"  {'Avg return after':<20} {losses['5d_ret'].mean():>+7.1f}% {losses['10d_ret'].mean():>+7.1f}% {losses['20d_ret'].mean():>+7.1f}%")
print(f"  {'Recovered (>0%)':<20} {losses['5d_ret'].gt(0).sum():>7}/{len(losses)} {losses['10d_ret'].gt(0).sum():>7}/{len(losses)} {losses['20d_ret'].gt(0).sum():>7}/{len(losses)}")
print(f"  {'Recovered (>+2%)':<20} {losses['5d_ret'].gt(2).sum():>7}/{len(losses)} {losses['10d_ret'].gt(2).sum():>7}/{len(losses)} {losses['20d_ret'].gt(2).sum():>7}/{len(losses)}")
print(f"  Avg max(10d): +{losses['max_10d'].mean():.1f}%")

# Net effect: if we held ALL losses 10d longer
print(f"\n  💰 If held ALL losses +10d: cumulative extra = {losses['10d_ret'].sum():+.1f}%")
print(f"  💰 If held ALL losses +20d: cumulative extra = {losses['20d_ret'].sum():+.1f}%")

# Individual
false_stops = losses[losses['10d_ret'] > 2].sort_values('pnl')
true_stops = losses[losses['10d_ret'] < -2].sort_values('pnl')
print(f"\n  🔴 False stops (loss, then +10d recovery >+2%): {len(false_stops)}/{len(losses)}")
for _, r in false_stops.iterrows():
    print(f"    {r['symbol']:<8} exit={r['exit_date']} PnL={r['pnl']:+.1f}%  +5d:{r['5d_ret']:+.1f}% +10d:{r['10d_ret']:+.1f}% max10d:{r['max_10d']:+.1f}%")

print(f"\n  🟢 True stops (loss, kept falling <-2%): {len(true_stops)}/{len(losses)}")
for _, r in true_stops.iterrows():
    print(f"    {r['symbol']:<8} exit={r['exit_date']} PnL={r['pnl']:+.1f}%  +5d:{r['5d_ret']:+.1f}% +10d:{r['10d_ret']:+.1f}%")

print(f"\n  🟡 Win exits, kept rising in 10d: {wins_exit['10d_ret'].gt(0).sum()}/{len(wins_exit)}")
print(f"    Avg +10d for wins: +{wins_exit['10d_ret'].mean():.1f}% — exited too early?")

# ═══════════════════════════════════════════
# ANALYSIS 2: Hold-for-N-days simulation
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("2. HOLD-FOR-N-DAYS SIMULATION (all 62 BUY events)")
print("="*70)

buys = trades[trades['action']=='BUY'].copy()
sim = []
for _, b in buys.iterrows():
    sym, buy_dt, buy_px = b['symbol'], b['date'], b['price']
    f = os.path.join(DAILY_K, f"{sym}.parquet")
    if not os.path.exists(f):
        continue
    s = pd.read_parquet(f)
    s['date'] = pd.to_datetime(s['date'])
    m = s[s['date'] == buy_dt]
    if m.empty:
        continue
    p = m.index[0]
    for hd in [3,5,7,10,15,20]:
        fp = min(p+hd, len(s)-1)
        ret = float((s.iloc[fp]['close']/buy_px - 1)*100)
        sim.append({'symbol': sym, 'buy_date': buy_dt.strftime('%Y%m%d'),
                    'hold_days': hd, 'ret': ret, 'win': ret > 0})

sd = pd.DataFrame(sim)
print(f"\n  {'Hold':<8} {'Win Rate':>10} {'Avg Ret':>10} {'Cum Ret':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for hd in [3,5,7,10,15,20]:
    sub = sd[sd['hold_days'] == hd]
    print(f"  {hd:>3}d     {sub['win'].mean()*100:>9.1f}%  {sub['ret'].mean():>+9.1f}%  {sub['ret'].sum():>+9.1f}%")

# ═══════════════════════════════════════════
# ANALYSIS 3: Score-based win rate
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("3. SCORE-BASED WIN RATE (MODEL REPLAY)")
print("="*70)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.repository import DataRepository
from models.catboost_model import CatBoostWrapper
from features.pipeline import FeaturePipeline, FACTOR_MAP
from features.processor import CrossSectionalProcessor
import yaml

with open('configs/pipeline_config.yaml') as f:
    cfg = yaml.safe_load(f)['pipeline']

repo = DataRepository()
model = CatBoostWrapper()
model_path = 'models/saved/catboost_single'
try:
    model.load(model_path)
    print("  Model loaded ✓")
except Exception as e:
    print(f"  Model load failed: {e}"); model = None

if model:
    factors = []
    for fc in cfg['features']:
        if fc['name'] == 'LabelGenerator': continue
        factors.append(FACTOR_MAP[fc['name']](**fc.get('params', {})))
    pipeline = FeaturePipeline(factors)
    processor = CrossSectionalProcessor(True, True)

    constituents = pd.read_parquet('data/local_lake/basics/csi_all_constituents.parquet')
    syms = constituents['symbol'].tolist()[:300]
    ben = repo.get_daily_data("sh.000300", start_date='2025-01-01', end_date='2026-04-30')

    all_observations = []

    for td in ['20260107', '20260204', '20260303']:
        feature_rows, valid = [], []
        for sym in syms:
            try:
                d = repo.get_daily_data(sym,
                    start_date=(pd.to_datetime(td)-pd.Timedelta(days=500)).strftime('%Y%m%d'),
                    end_date=td)
                if len(d) < 30: continue
                if 'date' in d.columns: d = d.set_index('date')
                if not ben.empty:
                    d['benchmark_close'] = ben.set_index('date')['close'].reindex(d.index, method='ffill')
                d['symbol'] = sym
                fv = pipeline.transform(d)
                if fv.empty: continue
                feature_rows.append(fv.iloc[-1])
                valid.append(sym)
            except: continue

        if not feature_rows: continue
        X = pd.DataFrame(feature_rows)
        for c in X.columns:
            if X[c].dtype == 'object': X[c] = pd.to_numeric(X[c], errors='coerce')
        drop_cols = ['open','high','low','close','volume','date','symbol']
        fc = [c for c in X.columns if c not in drop_cols]
        X = processor.process(X, fc)
        Xp = X[fc].select_dtypes(include=['number']).fillna(0)

        scores = model.predict(Xp)
        sdf = pd.DataFrame({'symbol': valid, 'score': scores}).sort_values('score', ascending=False)

        # Check actual future returns
        res = []
        for _, r in sdf.iterrows():
            try:
                d2 = repo.get_daily_data(r['symbol'], start_date=td, end_date='20260430')
                if len(d2) < 5: continue
                if 'date' in d2.columns: d2 = d2.set_index('date')
                f5 = d2['close'].iloc[min(5,len(d2)-1)]/d2['close'].iloc[0]-1
                f10 = d2['close'].iloc[min(10,len(d2)-1)]/d2['close'].iloc[0]-1
                res.append({'symbol': r['symbol'], 'score': r['score'],
                           'f5': f5*100, 'f10': f10*100, 'w5': f5>0, 'w10': f10>0})
            except: continue
        rdf = pd.DataFrame(res)
        if len(rdf) < 10: continue

        rdf['q'] = pd.qcut(rdf['score'], 4, labels=['Q1','Q2','Q3','Q4'])
        print(f"\n  ── {td} (n={len(rdf)}) ──")
        print(f"  {'Rank':<6} {'N':>5} {'Win5d':>8} {'Win10d':>8} {'Avg5d':>8} {'Avg10d':>8}")
        for q in ['Q4','Q3','Q2','Q1']:
            sub = rdf[rdf['q']==q]
            print(f"  {q:<6} {len(sub):>5} {sub['w5'].mean()*100:>7.1f}% {sub['w10'].mean()*100:>7.1f}% {sub['f5'].mean():>+7.1f}% {sub['f10'].mean():>+7.1f}%")

        top4 = rdf.head(4)
        r5_20 = rdf.iloc[4:20]
        r21_50 = rdf.iloc[20:50]
        print(f"  Top-4:   w10={top4['w10'].mean()*100:.0f}% avg={top4['f10'].mean():+.1f}%")
        print(f"  Rank5-20: w10={r5_20['w10'].mean()*100:.0f}% avg={r5_20['f10'].mean():+.1f}%")
        print(f"  Rank21-50: w10={r21_50['w10'].mean()*100:.0f}% avg={r21_50['f10'].mean():+.1f}%")

print("\n" + "="*70)
print("DONE")
