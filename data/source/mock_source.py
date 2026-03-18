from .base_source import BaseDataSource
import pandas as pd
import numpy as np

class MockDataSource(BaseDataSource):
    """
    Mock data source for testing.
    """
    def __init__(self):
        super().__init__("Mock")
        
    def get_stock_list(self):
        print("Mock: Generating stock list...")
        data = {
            'symbol': ['000001', '600000', '600519', '000002', '601398'],
            'name': ['MockBank1', 'MockBank2', 'MockLiquor', 'MockRealEstate', 'MockBank3']
        }
        return pd.DataFrame(data)

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        # print(f"Mock: Generating data for {symbol}...")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
            
        np.random.seed(int(symbol) if symbol.isdigit() else 0)
        start_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n)
        prices = start_price * (1 + returns).cumprod()
        
        data = {
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n)),
            'close': prices,
            'volume': np.random.randint(1000, 100000, n)
        }
        df = pd.DataFrame(data)
        # Set date as index to facilitate time series operations
        df = df.set_index('date')
        return df
