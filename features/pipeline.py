import pandas as pd
from .factors.base_factor import BaseFactor

class FeaturePipeline:
    """
    Manages the feature engineering process.
    """
    def __init__(self, factors):
        self.factors = factors

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
        
        # Usually we want features + labels aligned
        return all_data.dropna() # Drop rows with NaN (due to rolling/shift)
