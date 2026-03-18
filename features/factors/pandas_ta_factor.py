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
        # We need a copy to avoid SettingWithCopyWarning
        df_ta = df.copy()
        
        # Ensure index is datetime for pandas-ta
        if not isinstance(df_ta.index, pd.DatetimeIndex):
            if 'date' in df_ta.columns:
                df_ta = df_ta.set_index('date')
                df_ta.index = pd.to_datetime(df_ta.index)
        
        try:
            if self.strategy_mode.lower() == "all":
                df_ta.ta.strategy("all")
            elif self.strategy_mode.lower() == "common":
                df_ta.ta.strategy("common")
            elif self.strategy_mode.lower() == "custom" and self.features:
                # To support custom list, we still calculate 'all' or 'common' and then filter
                # Or we can parse the list and call specific functions, but that's complex mapping.
                # Easiest way: calculate 'all' and drop others. Efficient way: only call needed.
                # For now, let's calculate 'all' and then filter, assuming we want flexibility.
                # Optimization: Many pandas-ta indicators can be called directly.
                # But mapping "BBU_5_2.0" to ta.bbands(...) is hard without a parser.
                # So we stick to "all" strategy then filter.
                df_ta.ta.strategy("all")
            else:
                # Custom default small set
                df_ta.ta.sma(length=5, append=True)
                df_ta.ta.sma(length=20, append=True)
                df_ta.ta.rsi(length=14, append=True)
                df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
                df_ta.ta.bbands(length=20, append=True)
                
        except Exception as e:
            try:
                if self.strategy_mode.lower() in ["all", "common", "custom"]:
                    df_ta.ta.sma(length=10, append=True)
                    df_ta.ta.rsi(length=14, append=True)
                    df_ta.ta.macd(append=True)
                    df_ta.ta.kdj(append=True)
                    df_ta.ta.atr(append=True)
                    df_ta.ta.obv(append=True)
                    df_ta.ta.bbands(append=True)
            except:
                pass
            
        # Return only the newly created feature columns
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # If custom features are specified, filter them
        if self.strategy_mode.lower() == "custom" and self.features:
            # Check which requested features are actually in the dataframe
            available_features = [f for f in self.features if f in df_ta.columns]
            return df_ta[available_features]
            
        feature_cols = [col for col in df_ta.columns if col not in original_cols]
        
        return df_ta[feature_cols]
