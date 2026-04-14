import pandas as pd
import numpy as np

class DynamicFilter:
    """
    Filters out 'zombie' stocks (low volume/turnover) and newly listed stocks
    based on their time-series data dynamically before they are fed into the model.

    Zombie stocks (僵尸股) are stocks with extremely low trading activity, often manipulated
    or abandoned by the market. They can cause model overfitting to artificial price movements.

    Newly listed stocks (次新股) often have highly volatile prices due to market sentiment
    rather than fundamentals, making them unsuitable for modeling.

    Example:
        >>> filter = DynamicFilter(min_avg_turnover=10_000_000, min_listed_days=120)
        >>> filtered_df = filter.filter(stock_df)  # For a single stock's time-series
    """
    def __init__(self, min_avg_turnover=10000000, min_listed_days=120):
        """
        Args:
            min_avg_turnover (float): Minimum 20-day average turnover (成交额, in RMB).
                                      Default 10 million RMB. Stocks below this threshold
                                      are considered "zombie" and filtered out.
            min_listed_days (int): Minimum number of trading days the stock must have been listed.
                                   Default 120 days (~6 months). Stocks listed less than this
                                   are considered "new" and filtered out.
        """
        self.min_avg_turnover = min_avg_turnover
        self.min_listed_days = min_listed_days
        
    def filter(self, df):
        """
        Filter the DataFrame for a single stock.
        Returns an empty DataFrame if the stock does not meet the criteria on the LAST day.
        Or filters out rows where criteria are not met.

        Args:
            df (pd.DataFrame): Time-series DataFrame for a SINGLE stock.
        """
        if df.empty:
            return pd.DataFrame()

        # Check if the data spans enough calendar days (not just row count)
        # This handles the case where we fetch a limited window but the stock has been listed long enough
        calendar_days_span = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
        if calendar_days_span < self.min_listed_days:
            # Not enough calendar days in fetched data to satisfy min_listed_days requirement
            # Even if the stock was listed before our fetch window, we don't have enough history
            return pd.DataFrame()
            
        # Estimate turnover (成交额) if 'amount' column doesn't exist
        # Baostock 'volume' is in shares. turnover = volume * close
        if 'amount' in df.columns:
            turnover = df['amount']
        else:
            turnover = df['volume'] * df['close']
            
        # Calculate 20-day moving average of turnover
        avg_turnover_20d = turnover.rolling(window=20).mean()
        
        # We want to keep rows where the stock has been listed for at least min_listed_days
        # AND its recent 20-day avg turnover > min_avg_turnover

        # Create a mask for valid rows
        # 1. Listed days condition: use actual calendar days since first data point
        # (This correctly handles cases where the stock was listed less than min_listed_days ago)
        days_since_first = (df.index - df.index[0]).days
        valid_history_mask = days_since_first >= self.min_listed_days

        # 2. Turnover condition
        valid_turnover_mask = avg_turnover_20d >= self.min_avg_turnover
        
        # Combine masks
        valid_mask = valid_history_mask & valid_turnover_mask
        
        # Apply mask
        filtered_df = df[valid_mask]
        
        return filtered_df

class CrossSectionalProcessor:
    """
    Handles cross-sectional feature preprocessing to denoise data and handle extreme values.

    Applied on merged DataFrame of ALL stocks right before model training/inference.
    For each date (cross-section), processes all stocks' features independently.

    Processing steps per date:
        1. MAD Clipping: Remove extreme outliers beyond 3σ (based on Median Absolute Deviation)
        2. Z-Score: Normalize to zero mean and unit variance

    This eliminates market-wide noise (Beta) and prevents extreme values from dominating.

    Example:
        >>> processor = CrossSectionalProcessor(use_mad_clip=True, use_zscore=True)
        >>> feature_cols = ['MACD_12_26_9', 'RSI_14', 'BBU_5_2.0_2.0']
        >>> processed_df = processor.process(features_df, feature_cols)
    """
    def __init__(self, use_mad_clip=True, use_zscore=True):
        self.use_mad_clip = use_mad_clip
        self.use_zscore = use_zscore
        
    def _mad_clip(self, series, n=3):
        """
        Apply Median Absolute Deviation (MAD) clipping.
        Clips values outside of [Median - n * 1.4826 * MAD, Median + n * 1.4826 * MAD].
        """
        median = series.median()
        mad = (series - median).abs().median()
        
        # If MAD is 0 (e.g. lots of identical values), fallback to standard deviation clipping
        if mad == 0:
            std = series.std()
            upper_bound = median + n * std
            lower_bound = median - n * std
        else:
            upper_bound = median + n * 1.4826 * mad
            lower_bound = median - n * 1.4826 * mad
            
        return series.clip(lower=lower_bound, upper=upper_bound)
        
    def _zscore(self, series):
        """
        Apply Z-Score normalization: (x - mean) / std.
        """
        std = series.std()
        if std == 0 or pd.isna(std):
            # If standard deviation is 0, set all to 0 to avoid division by zero
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std

    def process(self, df, feature_cols):
        """
        Process the dataframe cross-sectionally (grouping by date index).
        
        Args:
            df: DataFrame with DatetimeIndex (representing dates). It contains data for all stocks.
            feature_cols: List of column names to process.
        
        Returns:
            DataFrame: A copy of the DataFrame with processed features.
        """
        if not self.use_mad_clip and not self.use_zscore:
            return df
            
        print(f"Applying cross-sectional processing (MAD Clip: {self.use_mad_clip}, Z-Score: {self.use_zscore})...")
        
        # Work on a copy to avoid SettingWithCopyWarning
        processed_df = df.copy()
        
        # We group by the index (which is date)
        # Note: Depending on Pandas version, groupby with transform on multiple columns might be slow.
        # We process feature by feature
        
        for col in feature_cols:
            if col not in processed_df.columns:
                continue
                
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                continue
                
            # 1. MAD Clipping
            if self.use_mad_clip:
                processed_df[col] = processed_df.groupby(level=0)[col].transform(self._mad_clip)
                
            # 2. Z-Score Normalization
            # Skip Z-Score for columns with near-zero variance within any group
            # (these features have no cross-sectional discrimination power)
            if self.use_zscore:
                group_stds = processed_df.groupby(level=0)[col].std()
                # Only apply Z-Score if std > 1e-8 for at least one group
                if group_stds.max() > 1e-8:
                    processed_df[col] = processed_df.groupby(level=0)[col].transform(self._zscore)
                else:
                    # Set to 0 to indicate no discrimination power
                    processed_df[col] = 0.0

        return processed_df