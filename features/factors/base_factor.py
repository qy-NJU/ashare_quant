from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    """
    Abstract base class for feature engineering factors.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate(self, df):
        """
        Calculate factor based on input DataFrame.
        Args:
            df (pd.DataFrame): OHLCV data.
        Returns:
            pd.Series or pd.DataFrame: Calculated factor values.
        """
        pass
