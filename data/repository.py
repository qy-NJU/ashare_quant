import pandas as pd
import os
import sys

class DataRepository:
    """
    Central repository for accessing data.
    Loads data exclusively from the local Data Lake (Parquet files).
    No on-the-fly downloading to ensure stability and performance.

    Data directory structure:
        data/local_lake/
        ├── daily_k/           # Per-stock daily OHLCV data (*.parquet)
        ├── basics/            # Stock list and metadata
        ├── events/            # Event data (Dragon Tiger List, etc.)
        └── features/          # Feature cache (MD5-hashed by config)

    Example:
        >>> repo = DataRepository(cache_dir='data/local_lake')
        >>> df = repo.get_daily_data('sh.600000', '20230101', '20230131')
        >>> print(df.head())
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
        self.events_dir = os.path.join(self.cache_dir, 'events')
        self.financials_dir = os.path.join(self.cache_dir, 'financials')
        self.funds_dir = os.path.join(self.cache_dir, 'funds')

        if not os.path.exists(self.daily_k_dir):
            print(f"Warning: Local Data Lake directory '{self.daily_k_dir}' does not exist.")
            print("Please run `python scripts/sync_data.py` first to populate local data.")

    def get_stock_list(self):
        """
        Load stock list from local Data Lake.

        Returns:
            pd.DataFrame: Stock list with columns like 'symbol', 'code', 'name', 'list_date', etc.
                          Returns empty DataFrame if not found.
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

    def get_industry_mapping(self):
        """
        Load industry classification mapping from local Data Lake.
        
        Returns:
            dict: Mapping of symbol to industry name. Returns empty dict if not found.
        """
        cache_file = os.path.join(self.basics_dir, 'industry_map.parquet')
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                return dict(zip(df['symbol'], df['industry']))
            except Exception as e:
                print(f"Error reading local industry mapping: {e}")
                
        return {}

    def get_financial_data(self, symbol, year, quarter):
        """
        Load quarterly financial data (profit/growth merged) from local Data Lake.
        
        Args:
            symbol (str): Stock symbol
            year (int): Year
            quarter (int): Quarter (1-4)
            
        Returns:
            pd.DataFrame: Merged financial data. Empty if not found.
        """
        cache_file = os.path.join(self.financials_dir, f"fin_{symbol}_{year}_{quarter}.parquet")
        
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error reading local financial data: {e}")
                
        return pd.DataFrame()

    def get_fund_flow(self, symbol):
        """
        Load daily fund flow data for a stock from local Data Lake.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Fund flow data. Empty if not found.
        """
        cache_file = os.path.join(self.funds_dir, f"fund_{symbol}.parquet")
        
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error reading local fund flow data: {e}")
                
        return pd.DataFrame()

    def get_lhb_data(self, start_date=None, end_date=None):
        """
        Load Longhu Bang (Dragon Tiger List) event data from local Parquet Data Lake.

        Dragon Tiger List (龙虎榜) contains institutional and hot-money trading data
        for stocks with significant price movements or unusual volume.

        Args:
            start_date (str): Filter events on or after this date (YYYY-MM-DD).
            end_date (str): Filter events on or before this date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Event data with columns like '上榜日', '代码', '名称',
                          '龙虎榜净买额', '龙虎榜买入额', etc. Returns empty DataFrame if no data.
        """
        cache_file = os.path.join(self.events_dir, 'lhb_data.parquet')

        if not os.path.exists(cache_file):
            return pd.DataFrame()

        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                return df

            # Ensure '上榜日' (list date) is datetime - note: column is '上榜日' not '上榜日期'
            if '上榜日' in df.columns:
                df['上榜日'] = pd.to_datetime(df['上榜日'])
            elif '上榜日期' in df.columns:
                df['上榜日期'] = pd.to_datetime(df['上榜日期'])

            # Apply date filters
            date_col = '上榜日' if '上榜日' in df.columns else '上榜日期'
            if start_date:
                df = df[df[date_col] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df[date_col] <= pd.to_datetime(end_date)]

            return df
        except Exception as e:
            print(f"Failed to load local LHB data: {e}")
            return pd.DataFrame()

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        """
        Load historical daily data from the local Parquet Data Lake and slice it by date.

        This is the primary method for accessing stock price data. Data is stored as
        one Parquet file per stock in the local data lake.

        Args:
            symbol (str): Stock symbol with prefix, e.g. 'sh.600000', 'sz.000001', 'bj.830779'.
            start_date (str): Start date in YYYYMMDD format.
            end_date (str): End date in YYYYMMDD format.
            adjust (str): Deprecated parameter, kept for backward compatibility.

        Returns:
            pd.DataFrame: Daily OHLCV data with DatetimeIndex.
                          Columns: open, high, low, close, volume, amount (if available).
                          Returns empty DataFrame if symbol not found or date range invalid.

        Example:
            >>> repo.get_daily_data('sh.600000', '20230101', '20230131')
                         open   high    low  close    volume     amount
            date
            2023-01-03  11.50  11.80  11.45  11.78  12345678  145678900
            ...
        """
        cache_file = os.path.join(self.daily_k_dir, f"{symbol}.parquet")

        # Fix: 兼容本地文件名没有前缀的情况 (如 "600000.parquet" 而不是 "sh.600000.parquet")
        if not os.path.exists(cache_file):
            symbol_only = symbol.replace('sh.', '').replace('sz.', '').replace('bj.', '')
            cache_file = os.path.join(self.daily_k_dir, f"{symbol_only}.parquet")

        if not os.path.exists(cache_file):
            # We don't download on-the-fly anymore. Just return empty DataFrame.
            return pd.DataFrame()

        try:
            # Load full history from local parquet (very fast - typically < 100ms for 20 years of data)
            df = pd.read_parquet(cache_file)

            if df.empty:
                return df

            # Ensure date column is set as index for date-based slicing
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            # Convert string dates to datetime for slicing
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Slice the requested date range using DatetimeIndex
            sliced_df = df.loc[start_dt:end_dt]

            return sliced_df

        except Exception as e:
            print(f"Failed to load/slice local data for {symbol}: {e}")
            return pd.DataFrame()
