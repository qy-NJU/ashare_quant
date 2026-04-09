import pandas as pd
from .repository import DataRepository

class StockPoolManager:
    """
    Manages the stock universe filtering based on board, exchange, and risk conditions (e.g. ST).
    """
    def __init__(self, data_repo=None):
        self.data_repo = data_repo if data_repo else DataRepository()

    def _identify_exchange(self, code):
        if code.startswith(('60', '68', '900')):
            return 'sh'
        elif code.startswith(('00', '30', '200')):
            return 'sz'
        elif code.startswith(('8', '43', '83', '87')):
            return 'bj'
        return 'unknown'

    def _identify_board(self, code):
        if code.startswith('68'):
            return 'star'     # 科创板
        elif code.startswith('30'):
            return 'chinext'  # 创业板
        elif code.startswith(('60', '00')):
            return 'main'     # 主板
        elif code.startswith(('8', '43', '83', '87')):
            return 'bj'       # 北交所
        return 'other'

    def get_filtered_symbols(self, board=None, exchange=None, max_count=None, exclude_st=True):
        """
        Get a list of stock symbols filtered by board, exchange, and ST status.
        """
        df = self.data_repo.get_stock_list()
        
        if df.empty:
            print("Failed to fetch stock list from Local Data Lake.")
            return []

        initial_count = len(df)

        # Apply ST filter
        if exclude_st and 'name' in df.columns:
            # Exclude stocks containing 'ST', '*ST', or '退'
            st_mask = df['name'].str.contains('ST|退', case=False, na=False)
            df = df[~st_mask]
            print(f"StockPoolManager: Excluded {initial_count - len(df)} ST/Delisting stocks.")

        df['exchange'] = df['symbol'].apply(self._identify_exchange)
        df['board'] = df['symbol'].apply(self._identify_board)

        # Apply filters
        if exchange and exchange.lower() != 'all':
            df = df[df['exchange'] == exchange.lower()]
            
        if board and board.lower() != 'all':
            df = df[df['board'] == board.lower()]

        symbols = df['symbol'].tolist()
        
        if max_count and max_count > 0:
            symbols = symbols[:max_count]
            
        print(f"StockPoolManager: Found {len(symbols)} stocks matching Criteria (Board: {board}, Exchange: {exchange})")
        return symbols
