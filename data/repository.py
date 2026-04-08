import pandas as pd
import os
import sys

class DataRepository:
    """
    Central repository for accessing data.
    Loads data exclusively from the local Data Lake (Parquet files).
    No on-the-fly downloading to ensure stability and performance.
    """
    def __init__(self, sources=None, cache_dir='data/local_lake'):
        """
        Args:
            sources (list): Deprecated. Ignored in this version.
            cache_dir (str): Root directory of the local Data Lake.
        """
        self.cache_dir = cache_dir
        self.daily_k_dir = os.path.join(self.cache_dir, 'daily_k')
        self.basics_dir = os.path.join(self.cache_dir, 'basics')
        
        if not os.path.exists(self.daily_k_dir):
            print(f"Warning: Local Data Lake directory '{self.daily_k_dir}' does not exist.")
            print("Please run `python scripts/sync_data.py` first to populate local data.")

    def get_stock_list(self):
        """
        Load stock list from local Data Lake.
        """
        cache_file = os.path.join(self.basics_dir, 'stock_list.parquet')
        
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error reading local stock list: {e}")
                
        print(f"Stock list not found at {cache_file}.")
        print("Please run `python scripts/sync_data.py` to sync basic data.")
        return pd.DataFrame()

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        """
        Load historical daily data from the local Parquet Data Lake and slice it by date.
        """
        cache_file = os.path.join(self.daily_k_dir, f"{symbol}.parquet")
        
        if not os.path.exists(cache_file):
            # We don't download on-the-fly anymore. Just return empty DataFrame.
            # print(f"Warning: No local data found for {symbol}. Run sync_data.py to download.")
            return pd.DataFrame()
            
        try:
            # Load full history from local parquet (very fast)
            df = pd.read_parquet(cache_file)
            
            if df.empty:
                return df
                
            # Set index if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
            # Convert string dates to datetime for slicing
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Slice the requested date range
            sliced_df = df.loc[start_dt:end_dt]
            
            return sliced_df
            
        except Exception as e:
            print(f"Failed to load/slice local data for {symbol}: {e}")
            return pd.DataFrame()
