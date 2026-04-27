from .base_factor import BaseFactor
import pandas as pd
import numpy as np

class TechnicalFactors(BaseFactor):
    """
    Collection of technical analysis factors.
    """
    def __init__(self):
        super().__init__("TechnicalFactors")

    def calculate(self, df):
        # This implementation calculates multiple factors and returns a DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages
        result['ma5'] = df['close'].rolling(window=5).mean()
        result['ma20'] = df['close'].rolling(window=20).mean()
        
        # Momentum (Rate of Change)
        result['roc5'] = df['close'].pct_change(periods=5)
        
        # Volatility
        result['vol20'] = df['close'].pct_change().rolling(window=20).std()
        
        return result

class LabelGenerator(BaseFactor):
    """
    Generates labels for supervised learning (e.g., future return).
    """
    def __init__(self, horizon=5, target_type='regression', decay_weights=None):
        """
        Args:
            horizon (int): Number of days to look ahead.
            target_type (str): Type of label to generate.
                - 'regression': Continuous future return (e.g., 0.025 for 2.5% up).
                - 'binary': 1 if return > 0 else 0.
                - 'classification_3': 1 (up > 1%), 0 (flat), -1 (down < -1%).
                - 'excess_return_binary': 1 if stock return > benchmark return, else 0.
                - 'rank_pct': Cross-sectional percentage rank of returns (0.0 to 1.0). Best for rank:pairwise.
                - 'decay_weighted': Multi-horizon decay-weighted composite label.
            decay_weights (list): Weights for [1d, 3d, 5d, 7d] horizons. Default [0.4, 0.3, 0.2, 0.1].
        """
        super().__init__("LabelGenerator")
        self.horizon = horizon
        self.target_type = target_type
        self.decay_weights = decay_weights or [0.4, 0.3, 0.2, 0.1]
        self._horizons = [1, 3, 5, 7]

    def _compute_return(self, df, h):
        """Compute forward return over h days, using T+1 open as entry price."""
        if 'open' not in df.columns:
            future_close = df['close'].shift(-h)
            entry_price = df['close']
        else:
            future_close = df['close'].shift(-h)
            entry_price = df['open'].shift(-1)
        return (future_close / entry_price) - 1

    def calculate(self, df):
        if self.target_type == 'decay_weighted':
            composite = pd.Series(0.0, index=df.index)
            for w, h in zip(self.decay_weights, self._horizons):
                ret_h = self._compute_return(df, h)
                composite = composite + w * ret_h.fillna(0.0)
            # Keep label only where the farthest horizon return is valid (no lookahead leak)
            last_valid = self._compute_return(df, self._horizons[-1])
            label = composite.where(last_valid.notna(), np.nan)
            return label.rename(f'target_{self.horizon}d')

        # Calculate future return over horizon.
        # To avoid lookahead bias and align with T+1 open execution:
        # Trade is executed at T+1 open, and evaluated at T+horizon close.
        # Return = (Close_{T+horizon} / Open_{T+1}) - 1
        raw_return = self._compute_return(df, self.horizon)

        if self.target_type == 'regression':
            label = raw_return
        elif self.target_type == 'rank_pct':
            # Note: Groupby date ranking should ideally be done AFTER combining all stocks.
            # But since calculate() runs per stock, we temporarily return raw_return here.
            # The actual cross-sectional ranking MUST be done in runner.py before training.
            # We flag this by returning the raw_return and handling it globally later.
            label = raw_return
        elif self.target_type == 'binary':
            # 1 if positive return, 0 otherwise
            label = (raw_return > 0).astype(int)
            # Preserve NaNs from the shift
            label = label.where(raw_return.notna(), np.nan)
        elif self.target_type == 'classification_3':
            # 1 for > 1% up, -1 for < -1% down, 0 otherwise
            conditions = [
                (raw_return > 0.01),
                (raw_return < -0.01)
            ]
            choices = [1, -1]
            label = pd.Series(np.select(conditions, choices, default=0), index=df.index)
            label = label.where(raw_return.notna(), np.nan)
        elif self.target_type == 'excess_return_binary':
            # Expecting 'benchmark_close' to be merged into df before this step, or we approximate 
            # by requiring it. If not present, fallback to absolute return.
            if 'benchmark_close' in df.columns:
                future_bm_close = df['benchmark_close'].shift(-self.horizon)
                current_bm_close = df['benchmark_close']
                bm_return = (future_bm_close / current_bm_close) - 1
                excess_return = raw_return - bm_return
                label = (excess_return > 0).astype(int)
                label = label.where(raw_return.notna() & bm_return.notna(), np.nan)
            else:
                print("Warning: 'benchmark_close' not found. Falling back to absolute binary return.")
                label = (raw_return > 0).astype(int)
                label = label.where(raw_return.notna(), np.nan)
        else:
            raise ValueError(f"Unsupported target_type: {self.target_type}")
            
        return label.rename(f'target_{self.horizon}d')
