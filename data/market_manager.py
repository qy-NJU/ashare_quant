import baostock as bs
import pandas as pd
import os

class MarketDataManager:
    """
    Manages fetching and caching of Market Index data (e.g., HS300).
    """
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_index_daily(self, symbol="sh.000300", start_date="2020-01-01", end_date="2025-12-31"):
        """
        Fetch daily index data.
        """
        # Clean dates for filename
        s_clean = start_date.replace('-', '')
        e_clean = end_date.replace('-', '')
        cache_file = os.path.join(self.cache_dir, f"index_{symbol}_{s_clean}_{e_clean}.parquet")
        
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # print(f"Fetching Index Data {symbol} from Baostock...")
        lg = bs.login()
        
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        bs.logout()
        
        if not data_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col])
            
        df = df.set_index('date')
        
        # Save cache
        df.to_parquet(cache_file, engine='pyarrow')
        
        return df
