from .base_source import BaseDataSource
import baostock as bs
import pandas as pd

class BaostockSource(BaseDataSource):
    """
    Data source implementation using Baostock.
    Baostock is good for historical data and is quite stable.
    """
    _login_count = 0

    def __init__(self):
        super().__init__("Baostock")
        self._ensure_login()

    def __del__(self):
        self._ensure_logout()

    @classmethod
    def _ensure_login(cls):
        if cls._login_count == 0:
            lg = bs.login()
            if lg.error_code == '0':
                print("Baostock login success!")
            else:
                print(f"Baostock login failed: {lg.error_msg}")
        cls._login_count += 1

    @classmethod
    def _ensure_logout(cls):
        cls._login_count -= 1
        if cls._login_count == 0:
            bs.logout()
            print("Baostock logout success!")

    def _convert_symbol(self, symbol):
        """Convert standard symbol (e.g. 600000) to Baostock format (sh.600000)."""
        if symbol.startswith('sh.') or symbol.startswith('sz.') or symbol.startswith('bj.'):
            return symbol
        if symbol.startswith('6'):
            return f"sh.{symbol}"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"sz.{symbol}"
        elif symbol.startswith('4') or symbol.startswith('8'):
            return f"bj.{symbol}"
        return symbol

    def _unconvert_symbol(self, bs_symbol):
        """Convert Baostock format back to standard symbol."""
        if '.' in bs_symbol:
            return bs_symbol.split('.')[1]
        return bs_symbol

    def get_stock_list(self):
        print("Baostock: Fetching ALL stock list...")
        
        # To get ALL stocks in Baostock, we use query_all_stock(day)
        # We use a recent valid trading day
        rs = bs.query_all_stock(day="2024-03-01")
        
        all_stocks = []
        while (rs.error_code == '0') & rs.next():
            all_stocks.append(rs.get_row_data())
            
        if not all_stocks:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_stocks, columns=rs.fields)
        # Filter only A-shares (sh.6, sz.0, sz.3, bj.4, bj.8)
        # Exclude indices (sh.000, sz.399)
        df = df[df['code'].str.match(r'^(sh\.6|sz\.0|sz\.3|bj\.[48])')]
        
        # Rename and clean columns
        df = df.rename(columns={'code': 'symbol', 'code_name': 'name'})
        df['symbol'] = df['symbol'].apply(self._unconvert_symbol)
        
        return df[['symbol', 'name']]

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        # print(f"Baostock: Fetching daily data for {symbol}...")
        
        # Convert adjust format
        adjust_flag = "3" # Default no adjust
        if adjust == 'qfq':
            adjust_flag = "2"
        elif adjust == 'hfq':
            adjust_flag = "1"
            
        # Format dates (Baostock expects YYYY-MM-DD)
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}" if len(start_date) == 8 else start_date
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}" if len(end_date) == 8 else end_date
        
        bs_symbol = self._convert_symbol(symbol)
        
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume",
            start_date=start, end_date=end,
            frequency="d", adjustflag=adjust_flag
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # Convert types
        df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.set_index('date')
        return df

    def get_profit_data(self, symbol, year, quarter):
        """
        Fetch quarterly profit data.
        """
        bs_symbol = self._convert_symbol(symbol)

        # Query profit data (roe, net_profit_ratio, gross_profit_rate, net_profit, eps, return_on_equity, etc.)
        # Here we select a few key indicators
        rs = bs.query_profit_data(code=bs_symbol, year=year, quarter=quarter)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        return pd.DataFrame(data_list, columns=rs.fields)

    def get_operation_data(self, symbol, year, quarter):
        """
        Fetch quarterly operation data.
        """
        bs_symbol = self._convert_symbol(symbol)
        
        rs = bs.query_operation_data(code=bs_symbol, year=year, quarter=quarter)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        return pd.DataFrame(data_list, columns=rs.fields)

    def get_growth_data(self, symbol, year, quarter):
        """
        Fetch quarterly growth data.
        """
        bs_symbol = self._convert_symbol(symbol)
        
        rs = bs.query_growth_data(code=bs_symbol, year=year, quarter=quarter)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()
            
        return pd.DataFrame(data_list, columns=rs.fields)
