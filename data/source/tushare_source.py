from .base_source import BaseDataSource
import tushare as ts
import pandas as pd
import datetime

class TushareSource(BaseDataSource):
    """
    Data source implementation using Tushare.
    Requires a valid Tushare token.
    """
    def __init__(self, token=None):
        super().__init__("Tushare")
        if token is None:
            try:
                import sys
                import os
                # Add project root to sys.path if not there
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if project_root not in sys.path:
                    sys.path.append(project_root)
                from config import TUSHARE_TOKEN
                token = TUSHARE_TOKEN
            except ImportError:
                import os
                token = os.environ.get('TUSHARE_TOKEN', '')
        self.token = token
        
        if not self.token or self.token == "your_tushare_token_here":
            print("Warning: Valid TUSHARE_TOKEN not found. Tushare API will fail.")
            self.pro = None
        else:
            try:
                # set_token attempts to write to ~/.tushare/tk.csv or ~/tk.csv
                # We skip it if we can initialize pro_api directly.
                self.pro = ts.pro_api(self.token)
            except Exception as e:
                print(f"Warning: Failed to initialize Tushare API: {e}")
                self.pro = None

    def _convert_symbol(self, symbol):
        """Convert standard symbol (e.g. 600000) to Tushare format (600000.SH)."""
        if '.' in symbol and len(symbol.split('.')[1]) == 2:
            return symbol # Already in XXX.SZ format
        if symbol.startswith('sh.') or symbol.startswith('sz.') or symbol.startswith('bj.'):
            parts = symbol.split('.')
            return f"{parts[1]}.{parts[0].upper()}"
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            return f"{symbol}.SZ"
        elif symbol.startswith('4') or symbol.startswith('8'):
            return f"{symbol}.BJ"
        return symbol

    def _unconvert_symbol(self, ts_symbol):
        """Convert Tushare format (600000.SH) back to standard symbol."""
        if '.' in ts_symbol:
            return ts_symbol.split('.')[0]
        return ts_symbol

    def get_stock_list(self):
        print("Tushare: Fetching ALL stock list...")
        try:
            df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
            if not df.empty:
                df = df.rename(columns={'ts_code': 'symbol'})
                df['symbol'] = df['symbol'].apply(self._unconvert_symbol)
                return df[['symbol', 'name']]
            return pd.DataFrame()
        except Exception as e:
            print(f"Tushare API Error: {e}")
            return pd.DataFrame()

    def get_industry_mapping(self):
        try:
            df = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
            if not df.empty:
                df = df.rename(columns={'ts_code': 'symbol'})
                df['symbol'] = df['symbol'].apply(self._unconvert_symbol)
                return df[['symbol', 'industry']]
            return pd.DataFrame()
        except Exception as e:
            print(f"Tushare API Error: {e}")
            return pd.DataFrame()

    def get_daily_data_batch(self, symbols, start_date, end_date, adjust='qfq'):
        """
        Fetch daily K-line data for multiple symbols in a single batch.
        Returns a dictionary mapping standard symbols to their DataFrames.
        """
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
        ts_codes = [self._convert_symbol(s) for s in symbols]
        ts_code_str = ','.join(ts_codes)
        
        try:
            # 1. Fetch unadjusted daily data
            df_daily = self.pro.daily(ts_code=ts_code_str, start_date=start, end_date=end)
            if df_daily is None or df_daily.empty:
                return {}
                
            # 2. Fetch adj factors
            df_adj = self.pro.adj_factor(ts_code=ts_code_str, start_date=start, end_date=end)
            
            # 3. Merge
            if df_adj is not None and not df_adj.empty:
                df = pd.merge(df_daily, df_adj, on=['ts_code', 'trade_date'], how='left')
                # For qfq, we need the latest adj_factor in the fetched period for each stock
                latest_adj = df_adj.sort_values('trade_date').groupby('ts_code').last()[['adj_factor']].rename(columns={'adj_factor': 'latest_factor'})
                df = pd.merge(df, latest_adj, on='ts_code', how='left')
                
                # Calculate qfq
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col] * df['adj_factor'] / df['latest_factor']
                    df[col] = df[col].round(2)
            else:
                df = df_daily
                
            # 4. Format and split by symbol
            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            
            # Tushare 'volume' (vol) is in lots (100 shares), Baostock is in shares. Convert to shares.
            df['volume'] = df['volume'] * 100
            
            # Tushare 'amount' is in thousands RMB. Baostock is in RMB. Convert to RMB.
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 1000
            
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if 'amount' in df.columns:
                cols.append('amount')
            cols.append('ts_code')
            df = df[cols]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                
            result = {}
            for ts_code, group in df.groupby('ts_code'):
                std_symbol = self._unconvert_symbol(ts_code)
                group = group.drop(columns=['ts_code']).sort_values('date').set_index('date')
                result[std_symbol] = group
                
            return result
        except Exception as e:
            print(f"Tushare API Error in batch: {e}")
            return {}

    def get_daily_data(self, symbol, start_date, end_date, adjust='qfq'):
        """
        Fetch daily K-line data for a specific symbol.
        """
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        ts_code = self._convert_symbol(symbol)
        
        try:
            # Tushare pro.daily gets unadjusted data. 
            # ts.pro_bar can get adjusted data (qfq, hfq)
            df = ts.pro_bar(ts_code=ts_code, start_date=start, end_date=end, adj=adjust, api=self.pro)
            
            if df is None or df.empty:
                return pd.DataFrame()
                
            # Tushare pro_bar returns: ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
            # Map columns to standard format
            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')
            
            # Tushare 'volume' (vol) is in lots (100 shares), Baostock is in shares. Convert to shares.
            df['volume'] = df['volume'] * 100
            
            # Tushare 'amount' is in thousands RMB. Baostock is in RMB. Convert to RMB.
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 1000
            
            # Select required columns
            cols = ['open', 'high', 'low', 'close', 'volume']
            if 'amount' in df.columns:
                cols.append('amount')
            df = df[cols]
            
            # Convert numeric types
            for col in cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
        except Exception as e:
            print(f"Tushare API Error for {symbol}: {e}")
            return pd.DataFrame()

    def get_profit_data(self, symbol, year, quarter):
        # Tushare financial data requires period format YYYYMMDD
        # Calculate the period end date based on quarter
        month_day = {1: '0331', 2: '0630', 3: '0930', 4: '1231'}
        period = f"{year}{month_day[quarter]}"
        ts_code = self._convert_symbol(symbol)
        
        try:
            df = self.pro.fina_indicator(ts_code=ts_code, period=period)
            if not df.empty:
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    def get_operation_data(self, symbol, year, quarter):
        # Already included in fina_indicator mostly, returning empty to avoid redundant calls
        return pd.DataFrame()

    def get_growth_data(self, symbol, year, quarter):
        # Included in fina_indicator
        return pd.DataFrame()
