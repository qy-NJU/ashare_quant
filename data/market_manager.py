import pandas as pd
import os
from data.repository import DataRepository

class MarketDataManager:
    """
    Manages fetching and caching of Market Index data (e.g., HS300).
    Loads data exclusively from the local Data Lake.
    """
    def __init__(self, cache_dir='data/local_lake'):
        self.repo = DataRepository(cache_dir=cache_dir)

    def get_index_daily(self, symbol="sh.000300", start_date="2020-01-01", end_date="2025-12-31"):
        """
        Fetch daily index data from local repository.
        """
        # Start/End date are usually strings like '2020-01-01', replace with '20200101' if needed
        # Or just pass them as they are, repository can handle strings.
        df = self.repo.get_daily_data(symbol, start_date.replace('-', ''), end_date.replace('-', ''))
        return df
