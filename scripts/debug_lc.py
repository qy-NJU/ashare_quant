"""Quick debug: check lifecycle data and stock file formats"""
import pandas as pd
import os

TRADES = 'data/backtest/trades_20260101_20260430.csv'
DAILY_K = 'data/local_lake/daily_k'

trades = pd.read_csv(TRADES)
trades['date'] = pd.to_datetime(trades['date'])

# Build lifecycles
lifecycles = []
current_positions = {}
for _, row in trades.sort_values('date').iterrows():
    sym = row['symbol']
    if row['action'] == 'BUY':
        if sym not in current_positions:
            current_positions[sym] = {'qty': 0, 'cost': 0, 'entry_dates': []}
        p = current_positions[sym]
        p['qty'] += row['qty']
        p['cost'] += row['qty'] * row['price']
        p['entry_dates'].append(row['date'])
    elif row['action'] == 'SELL' and sym in current_positions:
        p = current_positions[sym]
        if p['qty'] <= 0:
            continue
        sell_qty = min(row['qty'], p['qty'])
        avg_cost = p['cost'] / p['qty'] if p['qty'] > 0 else row['price']
        lifecycles.append({
            'symbol': sym,
            'entry_date': p['entry_dates'][0],
            'exit_date': row['date'],
            'avg_cost': avg_cost,
            'exit_price': row['price'],
            'pnl_pct': (row['price'] - avg_cost) / avg_cost * 100
        })
        p['qty'] -= sell_qty
        p['cost'] -= sell_qty * avg_cost
        if p['qty'] <= 0:
            del current_positions[sym]

df_lc = pd.DataFrame(lifecycles)
print(f"Lifecycles: {len(df_lc)}")
print(f"Sample:\n{df_lc.head(2)}")
print(f"exit_date type: {type(df_lc['exit_date'].iloc[0])}")

# Check a sample stock file
for sym in df_lc['symbol'].unique()[:3]:
    path = os.path.join(DAILY_K, f"{sym}.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"\n{sym}.parquet: {df.shape}, columns={list(df.columns)}")
        print(f"  date type: {df['date'].dtype}, sample: {df['date'].iloc[-1]}")
        # Check if exit_date matches
        exit_dt = pd.to_datetime(df_lc[df_lc['symbol']==sym]['exit_date'].iloc[0])
        match = df[df['date'] == exit_dt]
        print(f"  Looking for date={exit_dt}, match={len(match)} rows")
        if match.empty:
            # Check closest dates
            df['_diff'] = (df['date'] - exit_dt).abs()
            closest = df.nsmallest(3, '_diff')
            print(f"  Closest dates: {closest[['date','close']].to_string()}")
