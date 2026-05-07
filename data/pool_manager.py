import pandas as pd
import os
from .repository import DataRepository

class StockPoolManager:
    """
    Manages the stock universe filtering based on board, exchange, risk conditions (e.g. ST),
    and index constituents (CSI 300/500/1000).
    """
    def __init__(self, data_repo=None):
        self.data_repo = data_repo if data_repo else DataRepository()
        self._index_cache = {}

    def _load_index_constituents(self, index_name):
        if index_name in self._index_cache:
            return self._index_cache[index_name]

        basics_dir = os.path.join(self.data_repo.cache_dir, 'basics')
        path = os.path.join(basics_dir, f'{index_name}_constituents.parquet')

        if os.path.exists(path):
            df = pd.read_parquet(path)
            symbols = set(df['symbol'].tolist())
            self._index_cache[index_name] = symbols
            return symbols
        else:
            print(f"Warning: Index constituent file not found: {path}")
            return set()

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
            return 'star'
        elif code.startswith('30'):
            return 'chinext'
        elif code.startswith(('60', '00')):
            return 'main'
        elif code.startswith(('8', '43', '83', '87')):
            return 'bj'
        return 'other'

    def get_filtered_symbols(self, board=None, exchange=None, max_count=None,
                              exclude_st=True, indices=None):
        """
        Get a list of stock symbols filtered by board, exchange, ST status,
        and optionally by index constituents.

        Args:
            board: 'main', 'star', 'chinext', 'bj', or 'all'
            exchange: 'sh', 'sz', 'bj', or 'all'
            max_count: limit number of stocks returned
            exclude_st: filter out ST/*ST stocks
            indices: list of index names e.g. ['csi300', 'csi500', 'csi1000'],
                     or 'csi_all' for combined 300+500+1000.
                     When specified, only stocks in these indices are included.
        """
        df = self.data_repo.get_stock_list()

        if df.empty:
            print("Failed to fetch stock list from Local Data Lake.")
            return []

        initial_count = len(df)

        # Apply ST filter
        if exclude_st and 'name' in df.columns:
            st_mask = df['name'].str.contains('ST|退', case=False, na=False)
            df = df[~st_mask]
            print(f"StockPoolManager: Excluded {initial_count - len(df)} ST/Delisting stocks.")

        # Apply index constituent filter
        if indices:
            if indices == 'csi_all':
                index_set = self._load_index_constituents('csi_all')
            elif isinstance(indices, list):
                index_set = set()
                for idx_name in indices:
                    index_set.update(self._load_index_constituents(idx_name))
            elif isinstance(indices, str):
                index_set = self._load_index_constituents(indices)
            else:
                index_set = set()

            if index_set:
                before = len(df)
                df = df[df['symbol'].isin(index_set)]
                print(f"StockPoolManager: Index filter kept {len(df)}/{before} stocks.")

        df['exchange'] = df['symbol'].apply(self._identify_exchange)
        df['board'] = df['symbol'].apply(self._identify_board)

        # Apply exchange/board filters
        if exchange and exchange.lower() != 'all':
            df = df[df['exchange'] == exchange.lower()]

        if board and board.lower() != 'all':
            df = df[df['board'] == board.lower()]

        symbols = df['symbol'].tolist()

        if max_count and max_count > 0:
            symbols = symbols[:max_count]

        idx_info = f", Indices: {indices}" if indices else ""
        print(f"StockPoolManager: Found {len(symbols)} stocks (Board: {board}, Exchange: {exchange}{idx_info})")
        return symbols
