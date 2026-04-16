from .base_factor import BaseFactor
import pandas as pd
from data.repository import DataRepository
import os

class FundFlowFactor(BaseFactor):
    """
    Fetches daily individual fund flow data (Main/Super/Large/Medium/Small orders).
    """
    def __init__(self, name="FundFlowFactor", cache_dir='data/local_lake'):
        super().__init__(name)
        self.repo = DataRepository(cache_dir=cache_dir)

    def calculate(self, df):
        if 'symbol' not in df.columns:
            # print("Warning: 'symbol' column missing for FundFlowFactor")
            return pd.DataFrame(index=df.index)
            
        symbol = df['symbol'].iloc[0]
        
        fund_df = self.repo.get_fund_flow(symbol)
        
        if fund_df.empty:
            return pd.DataFrame(index=df.index)
            
        # Join with df index
        # fund_df index is date
        # df index is date
        
        try:
            # We select key fund flow metrics
            cols = ['main_net_ratio', 'super_net_ratio', 'main_net_inflow']
            available_cols = [c for c in cols if c in fund_df.columns]
            
            merged = df.join(fund_df[available_cols], how='left')
            result = merged[available_cols]
            
            # Fill NaNs with 0 (assuming no flow if missing)
            return result.fillna(0)
        except Exception as e:
            # print(f"FundFlow merge error: {e}")
            return pd.DataFrame(index=df.index)
