from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def select_stocks(self, date, data_loader, current_positions=None):
        """
        Select stocks for a given date.
        
        Args:
            date (str): Date in 'YYYYMMDD' format.
            data_loader: Data loader instance or module to fetch data.
            current_positions (dict): Current holdings, e.g., {'600000': 1000}.
            
        Returns:
            dict or list: 
                If dict: {symbol: target_weight}, e.g., {'600000': 0.5, '000001': 0.5}
                If list: [symbols], engine will default to equal weight.
        """
        pass
