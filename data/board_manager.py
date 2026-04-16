import pandas as pd
from data.repository import DataRepository

class BoardDataManager:
    """
    Manages fetching and caching of Industry and Concept mappings for stocks.
    Loads data exclusively from the local Data Lake.
    """
    def __init__(self, cache_dir='data/local_lake'):
        self.repo = DataRepository(cache_dir=cache_dir)
        self.industry_map = None

    def get_industry_mapping(self):
        """
        Returns a dict mapping stock symbols to their industry name.
        """
        if self.industry_map is not None:
            return self.industry_map

        self.industry_map = self.repo.get_industry_mapping()
        return self.industry_map
