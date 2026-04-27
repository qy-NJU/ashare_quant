import pandas as pd
import numpy as np
from .factors.base_factor import BaseFactor
from .factors.pandas_ta_factor import PandasTAFactor
from .factors.fundamental import BoardFactor
from .factors.financial import FinancialFactor
from .factors.fund_flow import FundFlowFactor
from .factors.market import MarketFactor
from .factors.subjective import SubjectiveFactor
from .factors.event_driven import EventFactor
from .factors.pattern import PatternFactor
from .factors.technical import LabelGenerator
from .factors.reversal import ReversalFactor

FACTOR_MAP = {
    "PandasTAFactor": PandasTAFactor,
    "BoardFactor": BoardFactor,
    "FinancialFactor": FinancialFactor,
    "FundFlowFactor": FundFlowFactor,
    "MarketFactor": MarketFactor,
    "SubjectiveFactor": SubjectiveFactor,
    "EventFactor": EventFactor,
    "PatternFactor": PatternFactor,
    "ReversalFactor": ReversalFactor,
    "LabelGenerator": LabelGenerator
}

class FeaturePipeline:
    """
    Manages the feature engineering process.
    """
    def __init__(self, factors):
        self.factors = factors

    def _add_temporal_features(self, df):
        """
        Add temporal derivatives for each numeric feature column:
          - _d5: 5-day absolute change (direction)
          - _r5: 5-day percentage change (velocity)
          - _s20: 20-day rolling standard deviation (stability)

        These give XGBoost the ability to learn "RSI went from 30 to 65"
        vs "RSI dropped from 80 to 65", which the raw snapshot alone cannot capture.
        """
        # Exclude raw price/volume and non-numeric columns from temporal encoding
        skip_cols = {'open', 'high', 'low', 'close', 'volume', 'amount', 'symbol', 'date'}
        feature_cols = [c for c in df.columns
                       if c not in skip_cols
                       and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

        temporal_parts = []
        for col in feature_cols:
            s = df[col]
            part = pd.DataFrame(index=df.index)
            shifted = s.shift(5)
            part[f'{col}_d5'] = s - shifted
            part[f'{col}_r5'] = s / shifted.replace(0, np.nan) - 1
            part[f'{col}_s20'] = s.rolling(20, min_periods=5).std()
            temporal_parts.append(part)

        if temporal_parts:
            temporal_features = pd.concat(temporal_parts, axis=1)
            df = pd.concat([df, temporal_features], axis=1)

        return df

    def transform(self, df):
        """
        Apply all factors to the data.
        """
        features_list = []
        for factor in self.factors:
            result = factor.calculate(df)
            if isinstance(result, pd.Series):
                result = result.to_frame()
            features_list.append(result)

        if not features_list:
            return pd.DataFrame()

        # Join all features
        all_features = pd.concat(features_list, axis=1)

        # Add original data if needed, or join on index
        # For simple integration, let's keep OHLCV + features
        all_data = pd.concat([df, all_features], axis=1)

        # Do NOT drop columns that are all NaN here, because different stocks might have different lengths
        # and dropping columns dynamically will cause feature mismatch between train and predict phases.
        # all_data = all_data.dropna(axis=1, how='all')

        # Drop duplicate columns (keep first occurrence)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]

        # Add temporal encodings for each numeric feature
        all_data = self._add_temporal_features(all_data)

        # Usually we want features + labels aligned
        # Do NOT drop all NaNs here, because some pandas-ta features might have NaNs
        # XGBoost can handle NaNs, or we can fill them later.
        # We only drop rows where the target label is NaN later in runner.py.
        return all_data
