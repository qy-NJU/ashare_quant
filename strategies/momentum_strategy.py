from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class MomentumStrategy(BaseStrategy):
    def __init__(self, name="Momentum20", period=20, top_n=5):
        super().__init__(name)
        self.period = period
        self.top_n = top_n

    def select_stocks(self, date, data_loader, stock_list=None):
        """
        Select stocks with highest return over the period.
        """
        if stock_list is None:
            # Fallback to fetching all stocks if not provided
            stock_list = data_loader.get_stock_list()
            if stock_list.empty:
                print("Warning: Stock list is empty.")
                return []
            stock_codes = stock_list['代码'].tolist()
        else:
            stock_codes = stock_list

        selected_stocks = []
        returns = []

        # Iterate over stocks (in a real system, vectorize this)
        # For demonstration, limit to first 10 stocks if list is large
        if len(stock_codes) > 10:
            print(f"Limiting to first 10 stocks for demonstration (out of {len(stock_codes)})")
            stock_codes = stock_codes[:10]

        for symbol in stock_codes:
            # Get data up to 'date'
            # Look back enough days to calculate momentum
            # date format YYYYMMDD
            try:
                # Calculate start date roughly
                dt = pd.to_datetime(date)
                start_dt = dt - pd.Timedelta(days=self.period * 2 + 10) # Buffer
                start_date_str = start_dt.strftime('%Y%m%d')
                
                df = data_loader.get_stock_daily(symbol, start_date=start_date_str, end_date=date)
                
                if len(df) < self.period:
                    continue
                
                # Calculate return: (Price_t / Price_{t-period}) - 1
                # Assuming '收盘' column exists
                if '收盘' not in df.columns:
                    continue
                    
                current_price = df.iloc[-1]['收盘']
                past_price = df.iloc[-self.period]['收盘']
                
                ret = (current_price / past_price) - 1
                returns.append((symbol, ret))
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        # Sort by return descending
        returns.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N
        selected_stocks = [x[0] for x in returns[:self.top_n]]
        
        print(f"Selected stocks for {date}: {selected_stocks}")
        return selected_stocks
