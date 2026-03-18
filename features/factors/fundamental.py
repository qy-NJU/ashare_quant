from .base_factor import BaseFactor
import pandas as pd
from data.board_manager import BoardDataManager

class BoardFactor(BaseFactor):
    """
    Appends Industry and Concept board information to the DataFrame.
    """
    def __init__(self, name="BoardFactor", encode_method="category"):
        """
        Args:
            name (str): Factor name.
            encode_method (str): How to encode categorical string. 
                                 "category" (for XGBoost native categorical support)
                                 or "label" (for Sklearn standard encoding).
        """
        super().__init__(name)
        self.encode_method = encode_method
        self.board_manager = BoardDataManager()
        
        # We need a shared label encoder state if using label encoding
        # For simplicity in 'category' mode, we just convert to pd.Categorical
        self.industry_map = self.board_manager.get_industry_mapping()

    def calculate(self, df):
        # We need to know which symbol this df belongs to.
        # Since our current pipeline passes OHLCV data per symbol without the symbol column,
        # we have a slight design issue.
        # Let's fix this by requiring the 'symbol' to be passed in, OR 
        # we assume 'symbol' is a column in df.
        
        df_result = pd.DataFrame(index=df.index)
        
        if 'symbol' not in df.columns:
            # If symbol is not in columns, we can't map it.
            # In a real system, the symbol should be part of the index or columns before feature engineering.
            # For this patch, we will return empty or default if symbol is missing.
            print("Warning: 'symbol' column not found in DataFrame. BoardFactor cannot be applied.")
            return df_result
            
        # Map industry
        df_result['industry'] = df['symbol'].map(self.industry_map).fillna("Unknown")
        
        if self.encode_method == "category":
            # Convert to pandas Categorical type, which XGBoost > 1.5 supports natively
            df_result['industry'] = df_result['industry'].astype('category')
        elif self.encode_method == "label":
            # Simple label encoding (factorize)
            df_result['industry'] = pd.factorize(df_result['industry'])[0]
            
        return df_result
