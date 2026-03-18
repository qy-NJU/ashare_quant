from .base_factor import BaseFactor
import pandas as pd
import numpy as np
from data.source.baostock_source import BaostockSource
import os

class FinancialFactor(BaseFactor):
    """
    Fetches and aligns quarterly financial data to daily price data.
    """
    def __init__(self, name="FinancialFactor", cache_dir='data/cache'):
        super().__init__(name)
        self.source = BaostockSource()
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def _fetch_quarterly_data(self, symbol, year, quarter):
        # We need a caching mechanism here because fetching per day is inefficient
        # Cache key: symbol_year_quarter
        cache_file = os.path.join(self.cache_dir, f"fin_{symbol}_{year}_{quarter}.parquet")
        
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)
            
        # Fetch from source
        # 1. Profit
        df_profit = self.source.get_profit_data(symbol, year, quarter)
        # 2. Growth
        df_growth = self.source.get_growth_data(symbol, year, quarter)
        
        # Merge if both exist
        if df_profit.empty and df_growth.empty:
            return pd.DataFrame()
        
        if df_profit.empty:
            df_merged = df_growth
        elif df_growth.empty:
            df_merged = df_profit
        else:
            # Both have 'code', 'pubDate', 'statDate'
            # We merge on common columns
            common_cols = list(set(df_profit.columns) & set(df_growth.columns))
            df_merged = pd.merge(df_profit, df_growth, on=common_cols, how='outer')
            
        if not df_merged.empty:
            df_merged.to_parquet(cache_file)
            
        return df_merged

    def calculate(self, df):
        if 'symbol' not in df.columns:
            print("Warning: 'symbol' column missing for FinancialFactor")
            return pd.DataFrame(index=df.index)
            
        symbol = df['symbol'].iloc[0]
        dates = df.index
        start_year = dates[0].year
        end_year = dates[-1].year
        
        # Collect all quarterly data in the range (plus one year back for initial fill)
        fin_records = []
        
        for y in range(start_year - 1, end_year + 1):
            for q in range(1, 5):
                q_df = self._fetch_quarterly_data(symbol, y, q)
                if not q_df.empty:
                    # Rename columns to meaningful names
                    # roeAvg, netProfit, YOYNI (YOY Net Income)
                    # We select a few key ones
                    cols_to_keep = {}
                    if 'roeAvg' in q_df.columns: cols_to_keep['roeAvg'] = 'fin_roe'
                    if 'netProfit' in q_df.columns: cols_to_keep['netProfit'] = 'fin_net_profit'
                    if 'YOYNI' in q_df.columns: cols_to_keep['YOYNI'] = 'fin_yoy_ni'
                    if 'pubDate' in q_df.columns: cols_to_keep['pubDate'] = 'pub_date'
                    
                    if not cols_to_keep: continue
                    
                    record = q_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep).iloc[0].to_dict()
                    fin_records.append(record)
                    
        if not fin_records:
            return pd.DataFrame(index=df.index)
            
        fin_df = pd.DataFrame(fin_records)
        fin_df['pub_date'] = pd.to_datetime(fin_df['pub_date'])
        fin_df = fin_df.sort_values('pub_date')
        
        # Now we need to merge this into the daily dataframe based on pub_date
        # We use merge_asof logic: for each day in df, take the latest financial record published before that day
        
        # Create a temporary dataframe with date index from df
        temp_df = pd.DataFrame(index=df.index)
        temp_df['date_idx'] = temp_df.index
        
        # Use pandas merge_asof
        fin_df = fin_df.sort_values('pub_date')
        # Ensure numeric types
        for col in ['fin_roe', 'fin_net_profit', 'fin_yoy_ni']:
            if col in fin_df.columns:
                fin_df[col] = pd.to_numeric(fin_df[col], errors='coerce')

        merged = pd.merge_asof(temp_df, fin_df, left_on='date_idx', right_on='pub_date', direction='backward')
        
        # Return only the financial feature columns
        feature_cols = [c for c in merged.columns if c.startswith('fin_')]
        result = merged[feature_cols]
        result.index = df.index
        
        return result.fillna(0) # Fill NaNs (e.g. before first report) with 0
