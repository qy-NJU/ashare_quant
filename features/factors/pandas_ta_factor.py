from .base_factor import BaseFactor
import pandas as pd
import pandas_ta as ta
import warnings

# Suppress pandas-ta warnings about missing TA-Lib
warnings.filterwarnings("ignore")
try:
    ta.utils.verbose = False
except:
    pass

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
            # Note: newer versions of pandas-ta use 'study' instead of 'strategy'
            # Also, cores=0 is required to prevent multiprocessing conflicts when run inside a ProcessPoolExecutor
            if hasattr(df_ta.ta, 'study'):
                ta_func = df_ta.ta.study
            else:
                ta_func = df_ta.ta.strategy
                
            if self.strategy_mode.lower() == "all":
                ta_func("all", cores=0, verbose=False)
            elif self.strategy_mode.lower() == "common":
                ta_func("common", cores=0, verbose=False)
            elif self.strategy_mode.lower() == "custom" and self.features:
                ta_func("all", cores=0, verbose=False)
            else:
                # Custom default small set
                df_ta.ta.sma(length=5, append=True)
                df_ta.ta.sma(length=20, append=True)
                df_ta.ta.rsi(length=14, append=True)
                df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)
                df_ta.ta.bbands(length=20, append=True)
                
        except Exception as e:
            # print(f"PandasTAFactor Exception: {e}")
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
