import pandas as pd
import numpy as np
from .base_factor import BaseFactor

class PatternFactor(BaseFactor):
    """
    Identifies classic chart patterns (morphological stock picking) for the A-share market.
    Includes: Bullish MA Alignment, Box/Consolidation Breakout, First Drop Engulfing, MACD Divergence.
    """
    def __init__(self):
        super().__init__("PatternFactor")
        
    def calculate(self, df):
        # Pre-requisite: ensure we have basic technical indicators if they are missing
        # We will calculate them locally to ensure this factor is independent of pandas_ta ordering
        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        vol = df['volume']
        pre_close = df['pre_close'] if 'pre_close' in df.columns else close.shift(1)
        
        # Create a new DataFrame for results to avoid modifying input df
        result = pd.DataFrame(index=df.index)
        
        # 1. 均线多头排列 (Bullish MA Alignment)
        # Short MA > Medium MA > Long MA, and trend is upwards
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        result['pat_bullish_ma'] = (
            (ma5 > ma10) & 
            (ma10 > ma20) & 
            (ma20 > ma60) & 
            (ma20 > ma20.shift(3)) # MA20 is trending up
        ).astype(int)
        
        # 2. 平台/箱体突破 (Box/Consolidation Breakout)
        # The stock has been trading in a narrow range for 20 days, and today it breaks out with high volume
        max_20 = high.shift(1).rolling(20).max()
        min_20 = low.shift(1).rolling(20).min()
        vol_ma5 = vol.rolling(5).mean()
        
        # Consolidation condition: Max and Min within 15% range
        is_consolidating = (max_20 - min_20) / min_20 < 0.15
        
        # Breakout condition: Close > Max of past 20 days, and Volume > 1.5 * 5-day Avg Volume
        is_breaking_out = (close > max_20) & (vol > 1.5 * vol_ma5)
        
        result['pat_box_breakout'] = (is_consolidating & is_breaking_out).astype(int)
        
        # 3. 龙头首阴反包 (First Drop Engulfing)
        # T-2: Big surge (> 7%)
        surge_t2 = (close.shift(2) / pre_close.shift(2) - 1) > 0.07
        
        # T-1: Yin candle (Close < Open) and shrinking volume
        yin_t1 = (close.shift(1) < open_p.shift(1))
        shrink_vol_t1 = (vol.shift(1) < vol.shift(2))
        
        # T: Yang candle (Close > Open) engulfing T-1's high
        yang_t = (close > open_p)
        engulfing_t = (close > high.shift(1))
        
        result['pat_first_drop_engulf'] = (surge_t2 & yin_t1 & shrink_vol_t1 & yang_t & engulfing_t).astype(int)
        
        # 4. 红三兵 (Three White Soldiers)
        # Three consecutive days of Higher Highs, Higher Lows, and Higher Closes (Yang candles)
        # Day 1
        yang_t2 = (close.shift(2) > open_p.shift(2))
        # Day 2
        yang_t1 = (close.shift(1) > open_p.shift(1)) & (close.shift(1) > close.shift(2))
        # Day 3
        yang_t0 = (close > open_p) & (close > close.shift(1))
        
        # Not stretched too far (Total gain in 3 days < 15%)
        not_overbought = (close / pre_close.shift(2) - 1) < 0.15
        
        result['pat_red_3_soldiers'] = (yang_t2 & yang_t1 & yang_t0 & not_overbought).astype(int)
        
        # 5. MACD 底背离 (MACD Bullish Divergence - Simplified)
        # Price hits a new 20-day low, but MACD histogram does not hit a new 20-day low.
        # We calculate a simple MACD first
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        # Price new low in past 5 days vs past 20 days
        price_new_low = low.rolling(5).min() <= low.rolling(20).min()
        
        # MACD hist is higher than its 20-day minimum (momentum is not as bad as price)
        macd_not_new_low = macd_hist > macd_hist.rolling(20).min()
        
        # MACD is turning upwards
        macd_turning_up = macd_hist > macd_hist.shift(1)
        
        result['pat_macd_divergence'] = (price_new_low & macd_not_new_low & macd_turning_up).astype(int)
        
        return result