import pandas as pd
import numpy as np
from .base_factor import BaseFactor


class ReversalFactor(BaseFactor):
    """
    Detects reversal signals to prevent the model from chasing extreme momentum.

    Features:
      A. Consecutive limit-up/down counters (prevent 000586-type disasters)
      B. Short-term reversal effects (1d, 5d returns — classic academic anomaly)
      C. Volume-price divergence (trend exhaustion signals)
      D. Extreme overbought/oversold metrics (cross-sectionally comparable)
    """

    def __init__(self, name="ReversalFactor"):
        super().__init__(name)

    def _get_limit_prices(self, df, symbol_str):
        """Compute limit up/down prices for the stock."""
        if 'pre_close' in df.columns:
            pre_close = df['pre_close']
        else:
            pre_close = df['close'].shift(1)

        is_star_or_chinext = (
            str(symbol_str).startswith('sh.688') or
            str(symbol_str).startswith('sz.300')
        )
        limit_ratio = 1.20 if is_star_or_chinext else 1.10

        limit_up_price = np.round(pre_close * limit_ratio, 2)
        limit_down_price = np.round(pre_close * (2 - limit_ratio), 2)
        return pre_close, limit_up_price, limit_down_price, limit_ratio

    def calculate(self, df):
        result = pd.DataFrame(index=df.index)

        if 'pre_close' in df.columns:
            pre_close = df['pre_close']
        else:
            pre_close = df['close'].shift(1)

        symbol_str = ""
        if 'symbol' in df.columns and not df.empty:
            symbol_str = str(df['symbol'].iloc[0])

        pre_close_arr, limit_up_price, limit_down_price, limit_ratio = \
            self._get_limit_prices(df, symbol_str)

        # ================================================================
        # A. Consecutive limit-up / limit-down counters
        # ================================================================

        is_limit_up = df['close'] >= limit_up_price - 0.01
        is_limit_down = df['close'] <= limit_down_price + 0.01

        # Count consecutive limit-up days
        cons_up = is_limit_up.astype(int)
        cons_up_count = cons_up.groupby((~is_limit_up).cumsum()).cumsum()
        # Only keep the count on the limit-up day itself; carry forward for context
        result['rev_consecutive_limit_up'] = cons_up_count.where(is_limit_up, 0.0)

        # Count consecutive limit-down days
        cons_down = is_limit_down.astype(int)
        cons_down_count = cons_down.groupby((~is_limit_down).cumsum()).cumsum()
        result['rev_consecutive_limit_down'] = cons_down_count.where(is_limit_down, 0.0)

        # Cumulative gain during the limit-up streak (profit locked in by streak buyers)
        cumulative_gain = pd.Series(0.0, index=df.index)
        streak_start = None
        for i in range(len(df)):
            if is_limit_up.iloc[i]:
                if streak_start is None:
                    streak_start = i
                cumulative_gain.iloc[i] = (
                    df['close'].iloc[i] / df['close'].iloc[streak_start] - 1
                    if streak_start is not None and streak_start > 0
                    else df['close'].iloc[i] / df['close'].iloc[max(streak_start, 1)] - 1
                )
            else:
                streak_start = None
        result['rev_limit_up_cumulative_gain'] = cumulative_gain

        # Days since last non-limit-up (how long has the streak been going?)
        # 0 = broke today, 1 = first limit day, N = N consecutive limit days
        result['rev_limit_up_streak_length'] = cons_up_count.where(is_limit_up, 0.0)

        # ================================================================
        # B. Short-term reversal effects (academic anomaly)
        # ================================================================

        result['rev_return_1d'] = df['close'].pct_change(1)
        result['rev_return_5d'] = df['close'].pct_change(5)

        # Reversal intensity: negative correlation between past 5d and next 1d
        # Compute rolling correlation between daily returns and next-day returns
        daily_ret = df['close'].pct_change()
        roll_corr = daily_ret.rolling(20).apply(
            lambda x: x[:-1].corr(x.shift(-1)[:-1]) if len(x.dropna()) >= 10 else 0
        )
        result['rev_reversal_corr_20d'] = roll_corr.fillna(0)

        # ================================================================
        # C. Volume-price divergence (trend exhaustion)
        # ================================================================

        # Price up but volume shrinking over last 5 days
        vol_ma5 = df['volume'].rolling(5).mean()
        vol_ma10 = df['volume'].rolling(10).mean()
        price_change_5d = df['close'].pct_change(5)

        # Volume ratio: short-term / medium-term
        result['rev_vol_ratio_5_10'] = vol_ma5 / vol_ma10.replace(0, np.nan)

        # Divergence: price rising (+5d) but volume declining (5d MA < 10d MA)
        price_up = price_change_5d > 0.03  # up > 3%
        vol_shrinking = vol_ma5 < vol_ma10 * 0.85  # volume < 85% of 10d avg
        result['rev_vol_price_divergence'] = (price_up & vol_shrinking).astype(int)

        # Volume climax: today's volume / 20d avg volume
        vol_ma20 = df['volume'].rolling(20).mean()
        result['rev_vol_climax_ratio'] = df['volume'] / vol_ma20.replace(0, np.nan)

        # Turnover rate estimation (if not already present)
        if 'amount' in df.columns:
            turnover = df['amount']
        else:
            turnover = df['volume'] * df['close']
        turnover_ma20 = turnover.rolling(20).mean()
        result['rev_turnover_spike'] = turnover / turnover_ma20.replace(0, np.nan)

        # ================================================================
        # D. Extreme overbought / oversold metrics
        # ================================================================

        # RSI-like: 14-day average gain / average loss ratio
        delta = df['close'].diff()
        avg_gain = delta.clip(lower=0).rolling(14).mean()
        avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))
        result['rev_rsi_14'] = rsi_14

        # Days in extreme RSI zone (>75 = overbought, <25 = oversold)
        overbought = rsi_14 > 75
        oversold = rsi_14 < 25
        result['rev_rsi_overbought_days'] = (
            overbought.astype(int)
            .groupby((~overbought).cumsum())
            .cumsum()
            .where(overbought, 0.0)
        )
        result['rev_rsi_oversold_days'] = (
            oversold.astype(int)
            .groupby((~oversold).cumsum())
            .cumsum()
            .where(oversold, 0.0)
        )

        # Bollinger Band position (cross-sectionally comparable)
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        result['rev_bb_position'] = (df['close'] - bb_mid) / bb_std.replace(0, np.nan)

        # Distance from 52-week (250-day) high
        high_250 = df['close'].rolling(250, min_periods=50).max()
        result['rev_dist_from_52w_high'] = df['close'] / high_250.replace(0, np.nan) - 1

        # Distance from 52-week low
        low_250 = df['close'].rolling(250, min_periods=50).min()
        result['rev_dist_from_52w_low'] = df['close'] / low_250.replace(0, np.nan) - 1

        # Price acceleration (2nd derivative): is momentum accelerating or decelerating?
        mom_5d = df['close'].pct_change(5)
        result['rev_momentum_accel'] = mom_5d - mom_5d.shift(5)
        # Negative acceleration + high momentum = potential reversal
        result['rev_momentum_decel'] = (
            (result['rev_momentum_accel'] < -0.05) & (mom_5d > 0.10)
        ).astype(int)

        # ================================================================
        # E. Gap risk indicators
        # ================================================================

        # Gap up from previous close (potential exhaustion gap after a streak)
        gap_up_pct = (df['open'] - pre_close_arr) / pre_close_arr.replace(0, np.nan)
        result['rev_gap_up_after_streak'] = (
            (gap_up_pct > 0.03) & (result['rev_consecutive_limit_up'].shift(1).fillna(0) >= 2)
        ).astype(int)

        # Consecutive positive days (not necessarily limit-up, just up days)
        up_days = (df['close'] > df['close'].shift(1)).astype(int)
        cons_up_normal = up_days.groupby((up_days == 0).cumsum()).cumsum()
        result['rev_consecutive_up_days'] = cons_up_normal

        # Consecutive negative days
        down_days = (df['close'] < df['close'].shift(1)).astype(int)
        cons_down_normal = down_days.groupby((down_days == 0).cumsum()).cumsum()
        result['rev_consecutive_down_days'] = cons_down_normal

        return result
