import baostock as bs
import pandas as pd
import os

class BoardDataManager:
    """
    Manages fetching and caching of Industry and Concept mappings for stocks.
    Currently uses Baostock for Shenwan (SW) Industry classification.
    """
    def __init__(self, cache_dir='data/cache'):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        self.industry_map = None

    def get_industry_mapping(self):
        """
        Returns a dict mapping stock symbols to their industry name.
        """
        if self.industry_map is not None:
            return self.industry_map

        cache_file = os.path.join(self.cache_dir, 'industry_map.parquet')
        
        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
            self.industry_map = dict(zip(df['symbol'], df['industry']))
            return self.industry_map

        print("Fetching Industry Classification from Baostock...")
        lg = bs.login()
        if lg.error_code != '0':
            print("Baostock login failed.")
            return {}

        rs = bs.query_stock_industry()
        industry_data = []
        while (rs.error_code == '0') & rs.next():
            industry_data.append(rs.get_row_data())
            
        bs.logout()

        if not industry_data:
            return {}

        df = pd.DataFrame(industry_data, columns=rs.fields)
        
        # Clean symbol (e.g. sh.600000 -> 600000)
        df['symbol'] = df['code'].apply(lambda x: x.split('.')[1] if '.' in x else x)
        
        # Save to cache
        df[['symbol', 'industry']].to_parquet(cache_file, engine='pyarrow', index=False)
        
        self.industry_map = dict(zip(df['symbol'], df['industry']))
        return self.industry_map
