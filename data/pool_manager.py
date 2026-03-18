import pandas as pd
from .source.baostock_source import BaostockSource

class StockPoolManager:
    """
    Manages the stock universe filtering based on board and exchange.
    """
    def __init__(self, source=None):
        # We use BaostockSource since AkShare might fail in the current network environment
        # Note: Baostock query_all_stock returns all stocks but we will fetch HS300 for stability in demo
        # For full market, we should call query_all_stock(day)
        self.source = source if source else BaostockSource()

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

    def get_filtered_symbols(self, board=None, exchange=None, max_count=None):
        """
        Get a list of stock symbols filtered by board and exchange.
        """
        # We will use the existing get_stock_list() which fetches HS300 in BaostockSource
        # If you need FULL market, you can implement get_all_stocks() in BaostockSource.
        # For demonstration and stability, we'll work with the list returned.
        df = self.source.get_stock_list()
        
        if df.empty:
            print("Failed to fetch stock list for filtering.")
            # Fallback to some mock symbols if network fails completely
            df = pd.DataFrame({'symbol': ['600000', '300001', '688001', '000001']})

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
