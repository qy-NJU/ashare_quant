from .base_factor import BaseFactor
import pandas as pd
import pandas_ta as ta

class PandasTAFactor(BaseFactor):
    """
    Wrapper for pandas-ta library to compute common technical indicators easily.
    """
    def __init__(self, name="PandasTA", strategy="default", features=None):
        """
        Args:
            name (str): Factor group name.
            strategy (str): Which pandas-ta strategy to use.
                            - "all": computes ~130 indicators.
                            - "common": computes a standard set of indicators.
                            - "default": computes a small custom set (SMA, RSI, MACD, BBands).
                            - "custom": computes only features specified in the 'features' list.
            features (list): List of feature names to keep (used when strategy="custom").
        """
        super().__init__(name)
        self.strategy_mode = strategy
        self.features = features

    def calculate(self, df):
        df_ta = df.copy()

        if not isinstance(df_ta.index, pd.DatetimeIndex):
            if 'date' in df_ta.columns:
                df_ta = df_ta.set_index('date')
                df_ta.index = pd.to_datetime(df_ta.index)

        if self.strategy_mode.lower() == "all":
            df_ta.ta.study("all", append=True, cores=0)
        elif self.strategy_mode.lower() == "common":
            df_ta.ta.study("common", append=True, cores=0)
        elif self.strategy_mode.lower() == "curated":
            # Curated set of ~45 non-redundant indicators covering all categories.
            # Avoids the 300+ highly correlated features from "all" (e.g., SMA_10 ≈ EMA_10 ≈ WMA_10).
            # Trend (multi-scale, only SMA+EMA to avoid MA-type redundancy)
            for length in [5, 10, 20, 60]:
                df_ta.ta.sma(length=length, append=True)
                df_ta.ta.ema(length=length, append=True)
            # Momentum
            df_ta.ta.rsi(length=14, append=True)
            df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_ta.ta.stoch(k=14, d=3, append=True)
            df_ta.ta.cci(length=20, append=True)
            df_ta.ta.willr(length=14, append=True)
            df_ta.ta.psar(append=True)
            df_ta.ta.ao(append=True)
            # Volatility
            df_ta.ta.bbands(length=20, std=2, append=True)
            df_ta.ta.atr(length=14, append=True)
            df_ta.ta.kc(length=20, scalar=2, append=True)
            # Volume / Money Flow
            df_ta.ta.obv(append=True)
            df_ta.ta.cmf(length=20, append=True)
            # Trend Strength
            df_ta.ta.adx(length=14, append=True)
        elif self.strategy_mode.lower() == "custom" and self.features:
            df_ta.ta.study("all", append=True, cores=0)
        else:
            df_ta.ta.sma(length=5, append=True)
            df_ta.ta.sma(length=20, append=True)
            df_ta.ta.rsi(length=14, append=True)
            df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
            df_ta.ta.bbands(length=20, append=True)

        original_cols = ['open', 'high', 'low', 'close', 'volume']

        if self.strategy_mode.lower() == "custom" and self.features:
            available_features = [f for f in self.features if f in df_ta.columns]
            return df_ta[available_features]

        feature_cols = [col for col in df_ta.columns if col not in original_cols]

        return df_ta[feature_cols]