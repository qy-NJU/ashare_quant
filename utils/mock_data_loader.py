import pandas as pd
import numpy as np
import datetime

def get_stock_list(force_update=False):
    """
    Mock stock list.
    """
    data = {
        '代码': ['000001', '600000', '600519', '000002', '601398'],
        '名称': ['平安银行', '浦发银行', '贵州茅台', '万科A', '工商银行'],
        '最新价': [10.5, 7.2, 1800.0, 9.8, 4.5]
    }
    return pd.DataFrame(data)

def get_stock_daily(symbol, start_date='20200101', end_date=None, adjust='qfq', force_update=False):
    """
    Mock daily data.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    dates = pd.date_range(start=start_dt, end=end_dt, freq='B') # Business days
    n = len(dates)
    
    # Generate random walk price
    np.random.seed(int(symbol) if symbol.isdigit() else 0)
    start_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n) # Mean 0.05%, Vol 2%
    prices = start_price * (1 + returns).cumprod()
    
    data = {
        '日期': dates,
        '开盘': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        '收盘': prices,
        '最高': prices * (1 + np.random.uniform(0, 0.02, n)),
        '最低': prices * (1 - np.random.uniform(0, 0.02, n)),
        '成交量': np.random.randint(1000, 100000, n),
        '成交额': np.random.randint(10000, 1000000, n)
    }
    
    return pd.DataFrame(data)
