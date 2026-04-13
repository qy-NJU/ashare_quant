from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    """
    Abstract base class for all feature engineering factors.

    All custom factors must inherit from this class and implement the `calculate` method.

    Example:
        ```python
        class MyCustomFactor(BaseFactor):
            def __init__(self, param1=1.0):
                super().__init__("MyCustomFactor")
                self.param1 = param1

            def calculate(self, df):
                # df contains OHLCV data with columns: open, high, low, close, volume, etc.
                result = df['close'].pct_change()
                return result
        ```

    Note:
        - The input `df` has a DatetimeIndex.
        - Return can be a pd.Series or pd.DataFrame, indexed the same as `df`.
        - For multiple output columns, return a DataFrame; for single column, return a Series.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, df):
        """
        Calculate factor based on input DataFrame.

        Args:
            df (pd.DataFrame): OHLCV data with DatetimeIndex. May also contain
                               additional columns like 'symbol', 'benchmark_close', etc.

        Returns:
            pd.Series or pd.DataFrame: Calculated factor values, indexed same as `df`.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        pass
