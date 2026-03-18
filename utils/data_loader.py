import akshare as ak
import pandas as pd
import os
import datetime
from config import RAW_DATA_DIR

def get_stock_list(force_update=False):
    """
    Get all A-share stock list.
    
    Args:
        force_update (bool): Whether to force update from AkShare.
        
    Returns:
        pd.DataFrame: DataFrame containing stock list.
    """
    file_path = os.path.join(RAW_DATA_DIR, 'stock_list.csv')
    
    if os.path.exists(file_path) and not force_update:
        print(f"Loading stock list from {file_path}")
        return pd.read_csv(file_path, dtype={'代码': str})
    
    print("Fetching stock list from AkShare...")
    try:
        # stock_zh_a_spot_em returns all A-share stocks with real-time data
        df = ak.stock_zh_a_spot_em()
        # Rename columns for consistency if needed, but keeping original Chinese names is fine for now
        # Columns: 序号, 代码, 名称, 最新价, 涨跌幅, ...
        df.to_csv(file_path, index=False)
        print(f"Stock list saved to {file_path}")
        return df
    except Exception as e:
        print(f"Error fetching stock list: {e}")
        return pd.DataFrame()

def get_stock_daily(symbol, start_date='20200101', end_date=None, adjust='qfq', force_update=False):
    """
    Get daily historical data for a stock.
    
    Args:
        symbol (str): Stock symbol (e.g., '600000').
        start_date (str): Start date in 'YYYYMMDD' format.
        end_date (str): End date in 'YYYYMMDD' format. Defaults to today.
        adjust (str): Adjustment type ('qfq', 'hfq', '').
        force_update (bool): Whether to force update from AkShare.
        
    Returns:
        pd.DataFrame: DataFrame containing daily data.
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        
    file_path = os.path.join(RAW_DATA_DIR, f'{symbol}_{adjust}.csv')
    
    if os.path.exists(file_path) and not force_update:
        # Load local data
        df = pd.read_csv(file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        
        # Filter by date
        mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
        return df.loc[mask]
    
    print(f"Fetching daily data for {symbol} from AkShare...")
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
        if not df.empty:
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        
    print("Testing get_stock_list...")
    stocks = get_stock_list()
    print(stocks.head())
    
    if not stocks.empty:
        symbol = stocks.iloc[0]['代码']
        print(f"\nTesting get_stock_daily for {symbol}...")
        daily_data = get_stock_daily(symbol, start_date='20230101')
        print(daily_data.head())
