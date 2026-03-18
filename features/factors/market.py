from .base_factor import BaseFactor
import pandas as pd
import numpy as np
from data.market_manager import MarketDataManager

class MarketFactor(BaseFactor):
    """
    Generates global market features and broadcasts them to individual stocks.
    Features include:
    - Index return / volatility (e.g. HS300)
    - Market sentiment (Limit Up/Down counts - computed from cross-section)
    """
    def __init__(self, name="MarketFactor", index_symbol="sh.000300"):
        super().__init__(name)
        self.index_symbol = index_symbol
        self.market_manager = MarketDataManager()
        self.index_df = None

    def prepare_index_data(self, start_date, end_date):
        """
        Pre-fetch index data for the period.
        """
        self.index_df = self.market_manager.get_index_daily(
            symbol=self.index_symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        if not self.index_df.empty:
            # Pre-calculate index features
            self.index_df['idx_ret'] = self.index_df['close'].pct_change()
            self.index_df['idx_ma20'] = self.index_df['close'].rolling(20).mean()
            # Distance from MA20 (Trend indicator)
            self.index_df['idx_trend'] = (self.index_df['close'] / self.index_df['idx_ma20']) - 1
            
            # Volatility
            self.index_df['idx_vol20'] = self.index_df['idx_ret'].rolling(20).std()

    def calculate(self, df, global_context=None):
        """
        Args:
            df (pd.DataFrame): Individual stock data.
            global_context (dict): Optional context containing cross-sectional stats 
                                   (e.g., limit_up_counts).
        """
        # 1. Merge Index Features (Broadcasting)
        if self.index_df is None:
            return pd.DataFrame(index=df.index)
            
        # Join based on date index
        # We only want specific columns
        features = ['idx_ret', 'idx_trend', 'idx_vol20']
        
        # Safe join
        try:
            # We use merge_asof or join. Since index is datetime, join is easiest.
            # We assume df index is datetime
            merged = df.join(self.index_df[features], how='left')
            result = merged[features]
        except Exception as e:
            # print(f"MarketFactor merge failed: {e}")
            result = pd.DataFrame(index=df.index, columns=features)
            
        # 2. Merge Cross-Sectional Stats (e.g. Limit Up Counts)
        # These must be computed outside and passed in via global_context
        # OR we compute them if we have access to the full panel.
        # Since 'calculate' runs per stock, we rely on 'global_context' injected by runner.
        
        if global_context and 'daily_stats' in global_context:
            stats_df = global_context['daily_stats']
            # Join stats
            stat_cols = ['limit_up_count', 'limit_down_count']
            try:
                merged_stats = df.join(stats_df[stat_cols], how='left')
                result = pd.concat([result, merged_stats[stat_cols]], axis=1)
            except:
                pass
                
        return result.fillna(0)
