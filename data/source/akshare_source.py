from .base_source import BaseDataSource
import akshare as ak
import pandas as pd
import datetime

class AkShareSource(BaseDataSource):
    """
    Data source implementation using AkShare.
    """
    def __init__(self):
        super().__init__("AkShare")

    def get_stock_list(self):
        try:
            print("AkShare: Fetching stock list...")
            df = ak.stock_zh_a_spot_em()
            # Standardize columns
            # Raw columns: 序号, 代码, 名称, 最新价, 涨跌幅, ...
            df = df.rename(columns={'代码': 'symbol', '名称': 'name'})
            return df[['symbol', 'name']]
        except Exception as e:
            print(f"AkShare Error: {e}")
            return pd.DataFrame(columns=['symbol', 'name'])

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        try:
            # print(f"AkShare: Fetching daily data for {symbol}...")
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
            
            if df.empty:
                return pd.DataFrame()
                
            # Standardize columns
            # Raw columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, ...
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"AkShare Error for {symbol}: {e}")
            return pd.DataFrame()

    def get_fund_flow(self, symbol):
        """
        Get daily individual fund flow data.
        Note: AkShare usually provides recent history for fund flow.
        """
        try:
            # Using stock_individual_fund_flow which provides history
            # Returns: 日期, 收盘价, 涨跌幅, 主力净流入-净额, 主力净流入-净占比, ...
            df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith("6") else "sz")
            
            if df.empty:
                return pd.DataFrame()
                
            # Rename columns
            rename_map = {
                '日期': 'date',
                '主力净流入-净额': 'main_net_inflow',
                '主力净流入-净占比': 'main_net_ratio',
                '超大单净流入-净额': 'super_net_inflow',
                '超大单净流入-净占比': 'super_net_ratio',
                '大单净流入-净额': 'large_net_inflow',
                '大单净流入-净占比': 'large_net_ratio',
                '中单净流入-净额': 'medium_net_inflow',
                '中单净流入-净占比': 'medium_net_ratio',
                '小单净流入-净额': 'small_net_inflow',
                '小单净流入-净占比': 'small_net_ratio'
            }
            df = df.rename(columns=rename_map)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Keep relevant columns
            keep_cols = [c for c in rename_map.values() if c != 'date']
            # Convert to numeric
            for col in keep_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df[keep_cols]
            
        except Exception as e:
            # print(f"AkShare Fund Flow Error for {symbol}: {e}")
            return pd.DataFrame()
