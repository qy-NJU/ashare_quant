import pandas as pd
import numpy as np
from .base_factor import BaseFactor
from data.repository import DataRepository

class EventFactor(BaseFactor):
    """
    Event-driven factors.
    Extracts events like Longhu Bang (Dragon Tiger List) and aligns them to daily K-line features.
    """
    def __init__(self):
        super().__init__("EventFactor")
        self.repo = DataRepository()
        
        # We load all LHB data into memory once since it's an event table, 
        # and we'll query it during calculate() for each stock
        self.lhb_df = self.repo.get_lhb_data()
        
        if not self.lhb_df.empty:
            # Clean and prepare LHB data
            # AkShare columns typically include: '上榜日期', '代码', '名称', '龙虎榜净买额', '买入额', '卖出额', '换手率', '市场总成交额', etc.
            self.lhb_df['date'] = pd.to_datetime(self.lhb_df['上榜日期'])
            
            # Format symbol to match Baostock format ('sh.600000', 'sz.000001')
            def format_symbol(code):
                code_str = str(code).zfill(6)
                if code_str.startswith('6'):
                    return f"sh.{code_str}"
                elif code_str.startswith('0') or code_str.startswith('3'):
                    return f"sz.{code_str}"
                elif code_str.startswith('8') or code_str.startswith('4'):
                    return f"bj.{code_str}"
                return code_str
                
            self.lhb_df['symbol'] = self.lhb_df['代码'].apply(format_symbol)
            
            # Index by symbol and date for fast lookup
            self.lhb_df = self.lhb_df.set_index(['symbol', 'date']).sort_index()

    def calculate(self, df):
        # If no LHB data is available or df is empty or 'symbol' not in df.columns, return NaN columns
        result = pd.DataFrame(index=df.index)
        
        if self.lhb_df.empty or df.empty or 'symbol' not in df.columns:
            result['evt_is_lhb'] = 0
            result['evt_lhb_net_buy_ratio'] = np.nan
            result['evt_lhb_buy_ratio'] = np.nan
            return result
            
        symbol = df['symbol'].iloc[0]
        
        # Initialize default values
        result['evt_is_lhb'] = 0
        result['evt_lhb_net_buy_ratio'] = np.nan
        result['evt_lhb_buy_ratio'] = np.nan
        
        # If this symbol has never been on LHB, just return
        if symbol not in self.lhb_df.index.get_level_values('symbol'):
            return result
            
        # Get LHB events for this specific stock
        stock_lhb = self.lhb_df.loc[symbol]
        
        # Merge LHB data into the result DataFrame based on date index
        # We expect df to have a DatetimeIndex
        
        # 1. Is on LHB today?
        # Check intersection of dates
        common_dates = df.index.intersection(stock_lhb.index)
        if len(common_dates) > 0:
            # Set flag to 1 for dates it was on LHB
            result.loc[common_dates, 'evt_is_lhb'] = 1
            
            # Calculate Net Buy Ratio (Net Buy / Total Market Turnover of the stock on that day)
            # If '龙虎榜净买额' and '市场总成交额' are available
            if '龙虎榜净买额' in stock_lhb.columns and '市场总成交额' in stock_lhb.columns:
                # Some values might be string with commas, need to clean
                try:
                    net_buy = pd.to_numeric(stock_lhb.loc[common_dates, '龙虎榜净买额'], errors='coerce')
                    total_turnover = pd.to_numeric(stock_lhb.loc[common_dates, '市场总成交额'], errors='coerce')
                    
                    # Prevent division by zero
                    ratio = np.where(total_turnover > 0, net_buy / total_turnover, 0)
                    result.loc[common_dates, 'evt_lhb_net_buy_ratio'] = ratio
                except Exception as e:
                    pass
                    
            # Calculate Total Buy Ratio (LHB Buy / Total Market Turnover)
            if '买入额' in stock_lhb.columns and '市场总成交额' in stock_lhb.columns:
                try:
                    buy_amount = pd.to_numeric(stock_lhb.loc[common_dates, '买入额'], errors='coerce')
                    total_turnover = pd.to_numeric(stock_lhb.loc[common_dates, '市场总成交额'], errors='coerce')
                    
                    ratio = np.where(total_turnover > 0, buy_amount / total_turnover, 0)
                    result.loc[common_dates, 'evt_lhb_buy_ratio'] = ratio
                except Exception as e:
                    pass

        return result