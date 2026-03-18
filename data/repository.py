import pandas as pd
import os
from .source.base_source import BaseDataSource

class DataRepository:
    """
    Central repository for accessing data.
    Handles Parquet caching and failover between sources.
    """
    def __init__(self, sources, cache_dir='data/cache'):
        """
        Args:
            sources (list): List of BaseDataSource instances, ordered by priority.
            cache_dir (str): Directory to store Parquet files.
        """
        self.sources = sources
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_stock_list(self):
        # We don't necessarily cache stock list as it's small, or we can use a simple CSV
        cache_file = os.path.join(self.cache_dir, 'stock_list.parquet')
        
        # Try load cache if it's recent (simplified: just load if exists)
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except:
                pass

        for source in self.sources:
            try:
                df = source.get_stock_list()
                if not df.empty:
                    df.to_parquet(cache_file, engine='pyarrow', index=False)
                    return df
            except Exception as e:
                print(f"Source {source.name} failed: {e}")
                continue
        print("All data sources failed to get stock list.")
        return pd.DataFrame()

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        """
        Try to load from Parquet cache first. If missing or incomplete, fetch from sources.
        For simplicity in this implementation, we cache by year-month or full request.
        Here we cache the specific request for demonstration.
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{adjust}.parquet")
        
        if os.path.exists(cache_file):
            try:
                # print(f"Loading {symbol} from Parquet cache...")
                df = pd.read_parquet(cache_file)
                # If index was saved as column during to_parquet, set it back
                if 'date' in df.columns:
                    df = df.set_index('date')
                return df
            except Exception as e:
                print(f"Failed to load cache for {symbol}: {e}")

        # Fetch from sources
        for source in self.sources:
            try:
                df = source.get_daily_data(symbol, start_date, end_date, adjust)
                if not df.empty:
                    # Save to cache
                    # Reset index to save 'date' column in Parquet
                    df_to_save = df.reset_index()
                    df_to_save.to_parquet(cache_file, engine='pyarrow', index=False)
                    return df
            except Exception as e:
                print(f"Source {source.name} failed for {symbol}: {e}")
                continue
                
        return pd.DataFrame()
