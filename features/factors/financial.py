from .base_factor import BaseFactor
import pandas as pd
import numpy as np
from data.repository import DataRepository
import os

class FinancialFactor(BaseFactor):
    """
    Fetches and aligns quarterly financial data to daily price data.
    """
    def __init__(self, name="FinancialFactor", cache_dir='data/local_lake'):
        super().__init__(name)
        self.repo = DataRepository(cache_dir=cache_dir)
            
    def _fetch_quarterly_data(self, symbol, year, quarter):
        # Read from local repository
        return self.repo.get_financial_data(symbol, year, quarter)

    def calculate(self, df):
        if 'symbol' not in df.columns:
            print("Warning: 'symbol' column missing for FinancialFactor")
            return pd.DataFrame(index=df.index)

        symbol = df['symbol'].iloc[0]
        dates = df.index
        start_year = dates[0].year
        end_year = dates[-1].year

        # Collect all quarterly data in the range (plus one year back for initial fill)
        fin_records = []

        for y in range(start_year - 1, end_year + 1):
            for q in range(1, 5):
                q_df = self._fetch_quarterly_data(symbol, y, q)
                if not q_df.empty:
                    cols_to_keep = {}
                    if 'roeAvg' in q_df.columns: cols_to_keep['roeAvg'] = 'fin_roe'
                    if 'netProfit' in q_df.columns: cols_to_keep['netProfit'] = 'fin_net_profit'
                    if 'YOYNI' in q_df.columns: cols_to_keep['YOYNI'] = 'fin_yoy_ni'
                    if 'npMargin' in q_df.columns: cols_to_keep['npMargin'] = 'fin_net_margin'
                    if 'gpMargin' in q_df.columns: cols_to_keep['gpMargin'] = 'fin_gross_margin'
                    if 'epsTTM' in q_df.columns: cols_to_keep['epsTTM'] = 'fin_eps_ttm'
                    if 'MBRevenue' in q_df.columns: cols_to_keep['MBRevenue'] = 'fin_revenue'
                    if 'totalShare' in q_df.columns: cols_to_keep['totalShare'] = 'fin_total_share'
                    if 'YOYEquity' in q_df.columns: cols_to_keep['YOYEquity'] = 'fin_yoy_equity'
                    if 'YOYAsset' in q_df.columns: cols_to_keep['YOYAsset'] = 'fin_yoy_asset'
                    if 'YOYEPSBasic' in q_df.columns: cols_to_keep['YOYEPSBasic'] = 'fin_yoy_eps'
                    if 'pubDate' in q_df.columns: cols_to_keep['pubDate'] = 'pub_date'

                    if not cols_to_keep: continue

                    record = q_df[list(cols_to_keep.keys())].rename(columns=cols_to_keep).iloc[0].to_dict()
                    fin_records.append(record)

        if not fin_records:
            return pd.DataFrame(index=df.index)

        fin_df = pd.DataFrame(fin_records)
        fin_df['pub_date'] = pd.to_datetime(fin_df['pub_date'])
        fin_df = fin_df.sort_values('pub_date')

        # Create a temporary dataframe with date index from df
        temp_df = pd.DataFrame(index=df.index)
        temp_df['date_idx'] = temp_df.index

        # Use pandas merge_asof
        fin_df = fin_df.sort_values('pub_date')
        numeric_cols = ['fin_roe', 'fin_net_profit', 'fin_yoy_ni', 'fin_net_margin',
                        'fin_gross_margin', 'fin_eps_ttm', 'fin_revenue', 'fin_total_share',
                        'fin_yoy_equity', 'fin_yoy_asset', 'fin_yoy_eps']
        for col in numeric_cols:
            if col in fin_df.columns:
                fin_df[col] = pd.to_numeric(fin_df[col], errors='coerce')

        merged = pd.merge_asof(temp_df, fin_df, left_on='date_idx', right_on='pub_date', direction='backward')
        merged.index = df.index  # align index to avoid comparison warnings

        # Compute valuation ratios from financial + price data (use .values to avoid index alignment issues)
        if 'close' in df.columns and 'fin_eps_ttm' in merged.columns:
            merged['fin_pe_ttm'] = df['close'].values / merged['fin_eps_ttm'].replace(0, np.nan).values
        if 'close' in df.columns and 'fin_total_share' in merged.columns and 'fin_revenue' in merged.columns:
            merged['fin_ps_ttm'] = (df['close'].values * merged['fin_total_share'].values) / merged['fin_revenue'].replace(0, np.nan).values

        # Return only the financial feature columns
        feature_cols = [c for c in merged.columns if c.startswith('fin_')]
        result = merged[feature_cols]
        result.index = df.index

        return result.fillna(0)
