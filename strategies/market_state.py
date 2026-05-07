import numpy as np
import pandas as pd


class MarketStateRecognizer:

    STATE_LABELS = ['trending_up', 'trending_down', 'ranging', 'crash', 'rebound']

    def __init__(self, index_symbol='sh.000300', ma_short=20, ma_long=60, ma_very_long=200,
                 vol_lookback=20, crash_threshold=-0.05, rebound_threshold=0.03):
        self.index_symbol = index_symbol
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.ma_very_long = ma_very_long
        self.vol_lookback = vol_lookback
        self.crash_threshold = crash_threshold
        self.rebound_threshold = rebound_threshold

        self.state_history = {}

    def recognize(self, date, data_repo):
        lookback_days = max(self.ma_very_long * 2, 500)
        start_date = (pd.to_datetime(date) - pd.Timedelta(days=lookback_days)).strftime('%Y%m%d')

        df = data_repo.get_daily_data(self.index_symbol, start_date=start_date, end_date=date)
        if df.empty or len(df) < self.ma_long:
            return 'ranging'

        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else None

        ma_short = close.rolling(self.ma_short).mean()
        ma_long = close.rolling(self.ma_long).mean()
        ma_very_long = close.rolling(self.ma_very_long).mean()

        current_close = close.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        current_ma_very_long = ma_very_long.iloc[-1] if len(close) >= self.ma_very_long else None

        returns = close.pct_change()
        daily_vol = returns.rolling(self.vol_lookback).std().iloc[-1] * np.sqrt(252)
        avg_vol = returns.rolling(self.vol_lookback * 5).std().iloc[-1] * np.sqrt(252)

        recent_return_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
        recent_return_20d = (close.iloc[-1] / close.iloc[-min(21, len(close))] - 1)

        drawdown_20d = (close.iloc[-min(21, len(close)):] / close.iloc[-min(21, len(close)):].cummax() - 1).min()

        trend_strength = 0
        if not pd.isna(current_ma_short) and not pd.isna(current_ma_long):
            if current_close > current_ma_short > current_ma_long:
                trend_strength = 2
            elif current_close > current_ma_short:
                trend_strength = 1
            elif current_close < current_ma_short < current_ma_long:
                trend_strength = -2
            elif current_close < current_ma_short:
                trend_strength = -1

        if current_ma_very_long is not None and not pd.isna(current_ma_very_long):
            above_ma200 = current_close > current_ma_very_long
        else:
            above_ma200 = True

        state = self._classify(
            trend_strength, recent_return_5d, recent_return_20d,
            drawdown_20d, daily_vol, avg_vol, above_ma200
        )

        self.state_history[date] = {
            'state': state,
            'trend_strength': trend_strength,
            'ret_5d': recent_return_5d,
            'ret_20d': recent_return_20d,
            'drawdown_20d': drawdown_20d,
            'vol_ratio': daily_vol / avg_vol if avg_vol > 0 else 1.0
        }

        return state

    def _classify(self, trend_strength, ret_5d, ret_20d, drawdown_20d, daily_vol, avg_vol, above_ma200):
        vol_ratio = daily_vol / avg_vol if avg_vol > 0 and not pd.isna(avg_vol) else 1.0

        if ret_5d <= self.crash_threshold or (drawdown_20d <= -0.08 and ret_5d < -0.03):
            return 'crash'

        if trend_strength >= 1 and ret_20d > 0.02 and above_ma200:
            return 'trending_up'
        elif trend_strength <= -1 and ret_20d < -0.02:
            return 'trending_down'

        if trend_strength == 0 and vol_ratio < 0.8:
            return 'ranging'

        if trend_strength == 0 and vol_ratio >= 0.8:
            if ret_5d > 0:
                return 'ranging'
            else:
                return 'ranging'

        return 'ranging'

    def get_recent_state(self, n=5):
        if not self.state_history:
            return 'ranging'
        recent_dates = sorted(self.state_history.keys())[-n:]
        states = [self.state_history[d]['state'] for d in recent_dates]
        return max(set(states), key=states.count)

    def should_reduce_position(self, date, data_repo):
        state = self.recognize(date, data_repo)

        if state == 'crash':
            return 0.80, "crash_emergency"

        if state == 'trending_down':
            df = data_repo.get_daily_data(
                self.index_symbol,
                start_date=(pd.to_datetime(date) - pd.Timedelta(days=10)).strftime('%Y%m%d'),
                end_date=date
            )
            if not df.empty and len(df) >= 4:
                below_ma200_count = 0
                close = df['close']
                ma200 = close.rolling(self.ma_very_long).mean()
                for i in range(-3, 1):
                    if i < 0 and abs(i) <= len(close):
                        if close.iloc[i] < ma200.iloc[i]:
                            below_ma200_count += 1
                if below_ma200_count >= 3:
                    return 0.85, "systemic_risk"

        return 0.0, "normal"
