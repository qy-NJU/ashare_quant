from abc import ABC, abstractmethod
import pandas as pd

class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_stock_list(self):
        """
        Get the list of all stocks.
        Returns:
            pd.DataFrame: Columns should include 'symbol', 'name'.
        """
        pass

    @abstractmethod
    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        """
        Get daily historical data.
        Args:
            symbol (str): Stock symbol.
            start_date (str): YYYYMMDD.
            end_date (str): YYYYMMDD.
            adjust (str): 'qfq', 'hfq', or None.
        Returns:
            pd.DataFrame: Standardized columns ['date', 'open', 'high', 'low', 'close', 'volume'].
        """
        pass
