import pandas as pd
import numpy as np
from .base_factor import BaseFactor

class SubjectiveFactor(BaseFactor):
    """
    Quantifies subjective short-term trading logics in the A-share market.
    Includes: Limit-up premium, turnover breakout, price-volume divergence, and weak-to-strong transitions.
    """
    def __init__(self):
        super().__init__("SubjectiveFactor")
        
    def calculate(self, df):
        # We need pre_close for accurate limit up/down calculations.
        # If pre_close is not in the data, we simulate it using close.shift(1)
        if 'pre_close' in df.columns:
            pre_close = df['pre_close']
        else:
            pre_close = df['close'].shift(1)
            
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns and not df.empty else ""
        is_star_or_chinext = str(symbol).startswith('sh.688') or str(symbol).startswith('sz.300')
        limit_ratio = 1.20 if is_star_or_chinext else 1.10
        
        # 1. 极限情绪因子 (Limit-up & Premium)
        limit_up_price = np.round(pre_close * limit_ratio, 2)
        limit_down_price = np.round(pre_close * (2 - limit_ratio), 2)
        
        df['sub_is_limit_up'] = (df['close'] >= limit_up_price - 0.01).astype(int)
        df['sub_is_limit_down'] = (df['close'] <= limit_down_price + 0.01).astype(int)
        
        # Premium: Yesterday's limit up stocks opening/high today
        df['sub_high_premium_tags'] = np.where(
            df['sub_is_limit_up'].shift(1) == 1,
            (df['high'] / pre_close) - 1,
            np.nan
        )
        
        # 2. 资金共识因子 (Turnover Breakout)
        # Estimate turnover if 'amount' is not available
        if 'amount' in df.columns:
            turnover = df['amount']
        else:
            turnover = df['volume'] * df['close']
            
        df['sub_turnover_rate_est'] = turnover / turnover.rolling(window=20, min_periods=5).mean()
        df['sub_vol_breakout_ratio'] = df['volume'] / df['volume'].rolling(window=5, min_periods=2).mean()
        df['sub_price_new_high_20'] = (df['close'] >= df['close'].rolling(window=20, min_periods=10).max()).astype(int)
        
        # 3. 风险预警因子 (Price-Volume Divergence)
        df['sub_upper_shadow_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / pre_close
        
        recent_gain = df['close'] / df['close'].shift(10) - 1
        df['sub_high_vol_stagnation'] = np.where(
            (recent_gain > 0.20) & (df['sub_vol_breakout_ratio'] > 2) & (df['close'] / pre_close - 1 < 0.02),
            1, 0
        )
        
        # 4. 预期差因子 (Weak to Strong)
        # Yesterday left a long upper shadow and high volume (Weak), today opened surprisingly high (Strong)
        weak_yesterday = (df['sub_upper_shadow_ratio'].shift(1) > 0.03) & (df['sub_vol_breakout_ratio'].shift(1) > 1.5)
        strong_today = df['open'] > pre_close * 1.02
        df['sub_weak_to_strong_signal'] = (weak_yesterday & strong_today).astype(int)
        
        # 5. 相对强弱因子 (Relative Strength)
        # Note: Since we compute this symbol by symbol, we expect the benchmark index to be merged in runner.py
        # as 'benchmark_close' before pipeline transformation.
        if 'benchmark_close' in df.columns:
            stock_return = df['close'] / pre_close - 1
            bench_pre_close = df['benchmark_close'].shift(1)
            bench_return = df['benchmark_close'] / bench_pre_close - 1
            df['sub_excess_return_hs300'] = stock_return - bench_return
            df['sub_excess_return_5d'] = (df['close'] / df['close'].shift(5)) - (df['benchmark_close'] / df['benchmark_close'].shift(5))
        else:
            df['sub_excess_return_hs300'] = np.nan
            df['sub_excess_return_5d'] = np.nan
            
        return df