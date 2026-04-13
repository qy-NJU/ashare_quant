from .base_model import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr


def calculate_ic(predictions, actual_returns):
    """
    Calculate Information Coefficient (IC) - Pearson correlation between predictions and returns.

    Args:
        predictions: array-like, model predictions
        actual_returns: array-like, actual future returns

    Returns:
        float: IC value (range -1 to 1)
    """
    # Remove NaN pairs
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(predictions[mask], actual_returns[mask])[0, 1]


def calculate_rank_ic(predictions, actual_returns):
    """
    Calculate Rank Information Coefficient (Rank IC) - Spearman correlation.

    More robust than IC as it uses rankings, less sensitive to outliers.

    Args:
        predictions: array-like, model predictions
        actual_returns: array-like, actual future returns

    Returns:
        float: Rank IC value (range -1 to 1)
    """
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 3:
        return np.nan
    return spearmanr(predictions[mask], actual_returns[mask]).correlation


def calculate_icir(ic_series):
    """
    Calculate IC Information Ratio (ICIR).

    IR = mean(IC) / std(IC)
    Measures the consistency of IC performance over time.

    Args:
        ic_series: array-like, IC values over multiple periods

    Returns:
        float: IR value (higher is better, >1.0 is good, >2.0 is excellent)
    """
    ic_array = np.array(ic_series)
    ic_array = ic_array[~np.isnan(ic_array)]
    if len(ic_array) == 0:
        return np.nan
    return ic_array.mean() / ic_array.std()


class XGBoostWrapper(BaseModel):
    """
    Wrapper for XGBoost supporting:
    - Incremental learning via xgb_model parameter
    - Early stopping to prevent overfitting
    - Training evaluation metrics (AUC, Error)
    - Native categorical feature support
    """
    def __init__(self, params=None, name="XGBoost"):
        super().__init__(name)
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'eta': 0.1,
            'verbosity': 0
        }
        self.booster = None
        # Store training history for analysis
        self.train_history = []

    def _prepare_data(self, X):
        """
        Prepare features for XGBoost: select numeric/category types and fill NaN.

        Args:
            X: Feature DataFrame

        Returns:
            xgb.DMatrix: Prepared DMatrix object
        """
        # Ensure categorical columns have proper category dtype
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['category']).columns:
            X_clean[col] = X_clean[col].cat.as_ordered()

        X_clean = X_clean.select_dtypes(include=['number', 'category']).fillna(0)
        return X_clean

    def _get_eval_metric(self):
        """Determine evaluation metric based on objective."""
        objective = self.params.get('objective', 'reg:squarederror')
        if 'rank:' in objective:
            return 'ndcg'  # Ranking uses NDCG
        elif 'binary' in objective:
            return 'auc'   # Binary classification uses AUC
        elif 'multi' in objective:
            return 'merror'  # Multi-class uses error rate
        else:
            return 'rmse'   # Regression uses RMSE

    def train(self, X, y, num_boost_round=50, groups=None, eval_X=None, eval_y=None,
              early_stopping_rounds=10, eval_ic=False):
        """
        Train XGBoost model with optional early stopping and IC evaluation.

        Args:
            X: Training features
            y: Training labels
            num_boost_round: Maximum number of boosting rounds (default: 50)
            groups: Group sizes for ranking (required when objective is rank:*)
            eval_X: Optional validation features for early stopping
            eval_y: Optional validation labels for early stopping
            early_stopping_rounds: Stop if no improvement after N rounds (default: 10, disabled if eval_X is None)
            eval_ic: If True, calculate IC/Rank IC after training (default: False)

        Returns:
            dict: Training result with metrics
        """
        X_clean = self._prepare_data(X)
        dtrain = xgb.DMatrix(X_clean, label=y, enable_categorical=True)

        if groups is not None:
            dtrain.set_group(groups)

        # Build evaluation list
        evals = [(dtrain, 'train')]
        dval = None

        # Add validation set for early stopping if provided
        if eval_X is not None and eval_y is not None:
            X_val_clean = self._prepare_data(eval_X)
            dval = xgb.DMatrix(X_val_clean, label=eval_y, enable_categorical=True)
            if groups is not None:
                # For ranking, we need to provide group sizes for validation too
                # Assume same group structure (this may need adjustment)
                dval.set_group(groups)
            evals.append((dval, 'eval'))

        # Determine if early stopping should be used
        use_early_stop = (dval is not None and early_stopping_rounds > 0)

        # Build training params with eval metric
        train_params = self.params.copy()
        eval_metric = self._get_eval_metric()
        if 'eval_metric' not in train_params:
            train_params['eval_metric'] = eval_metric

        # Train
        if use_early_stop:
            self.booster = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            best_iteration = self.booster.best_iteration
            best_score = self.booster.best_score
            print(f"[{self.name}] Training complete. Best iteration: {best_iteration}, "
                  f"Best {eval_metric}: {best_score:.4f}")
        else:
            self.booster = xgb.train(
                train_params,
                dtrain,
                num_boost_round=num_boost_round,
                verbose_eval=False
            )
            # Calculate training metric on full training set
            if eval_metric == 'auc':
                train_preds = self.booster.predict(dtrain)
                from sklearn.metrics import roc_auc_score
                train_score = roc_auc_score(y, train_preds)
            elif eval_metric == 'rmse':
                train_preds = self.booster.predict(dtrain)
                train_score = np.sqrt(np.mean((y - train_preds) ** 2))
            else:
                train_score = None

            print(f"[{self.name}] Training complete. Rounds: {num_boost_round}", end="")
            if train_score is not None:
                print(f", Train {eval_metric}: {train_score:.4f}")
            else:
                print()

        # Calculate IC metrics if requested (useful for regression/ranking objectives)
        ic_result = None
        if eval_ic and self.params.get('objective') in ['reg:squarederror', 'rank:pairwise']:
            train_preds = self.booster.predict(dtrain)
            ic_result = {
                'ic': calculate_ic(train_preds, y),
                'rank_ic': calculate_rank_ic(train_preds, y)
            }
            print(f"[{self.name}] IC Metrics: IC={ic_result['ic']:.4f}, Rank IC={ic_result['rank_ic']:.4f}")

        # Store training info
        self.train_history.append({
            'type': 'full',
            'rounds': num_boost_round,
            'metric': eval_metric,
            'ic_result': ic_result
        })

        return self.booster

    def partial_train(self, X, y, num_boost_round=10, groups=None):
        """
        Incrementally update the model using new data.

        Note: For incremental learning, we use a lower learning rate (eta * 0.5)
        and fewer rounds to avoid catastrophic forgetting.

        Args:
            X: New training features
            y: New training labels
            num_boost_round: Number of boosting rounds for this incremental update (default: 10)
            groups: Group sizes for ranking
        """
        X_clean = self._prepare_data(X)
        dtrain = xgb.DMatrix(X_clean, label=y, enable_categorical=True)

        if groups is not None:
            dtrain.set_group(groups)

        if self.booster is None:
            print(f"[{self.name}] No existing model found. Starting full train.")
            self.train(X, y, num_boost_round=num_boost_round, groups=groups)
            return

        # For incremental learning, we reduce learning rate to prevent overwriting
        # previous knowledge too quickly
        incremental_params = self.params.copy()
        original_eta = incremental_params.get('eta', 0.1)
        incremental_params['eta'] = original_eta * 0.5  # Halve the learning rate

        # Continue training from existing booster with reduced learning rate
        self.booster = xgb.train(
            incremental_params,
            dtrain,
            num_boost_round=num_boost_round,
            xgb_model=self.booster
        )
        print(f"[{self.name}] Incremental training complete. "
              f"(eta reduced to {incremental_params['eta']:.4f} for stability)")

        self.train_history.append({
            'type': 'incremental',
            'rounds': num_boost_round,
            'eta_used': incremental_params['eta']
        })

    def predict(self, X):
        """Predict using trained model."""
        if self.booster is None:
            raise ValueError("Model is not trained yet.")
        X_clean = self._prepare_data(X)
        dtest = xgb.DMatrix(X_clean, enable_categorical=True)
        return self.booster.predict(dtest)

    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance scores.

        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'

        Returns:
            dict: Feature name -> importance score
        """
        if self.booster is None:
            raise ValueError("Model is not trained yet.")

        scores = self.booster.get_score(importance_type=importance_type)
        return scores

    def evaluate_ic(self, X, y_true):
        """
        Evaluate model using Information Coefficient metrics.

        Calculates IC (Pearson) and Rank IC (Spearman) between predictions and actual returns.

        Args:
            X: Features for prediction
            y_true: Actual returns/labels (same length as predictions)

        Returns:
            dict: {'ic': float, 'rank_ic': float, 'ir': float or None}
        """
        predictions = self.predict(X)

        ic = calculate_ic(predictions, y_true)
        rank_ic = calculate_rank_ic(predictions, y_true)

        result = {
            'ic': ic,
            'rank_ic': rank_ic,
            'ir': None  # IR requires multiple periods, use evaluate_ic_series instead
        }

        return result

    def evaluate_ic_series(self, ic_data_list):
        """
        Calculate IR (IC Information Ratio) from a series of IC values.

        Args:
            ic_data_list: List of tuples (predictions, actual_returns) for multiple periods,
                          or list of IC values directly

        Returns:
            dict: {'ic_mean': float, 'ic_std': float, 'ir': float, 'ic_series': list}
        """
        # If input is list of (pred, actual) pairs, calculate IC for each
        if len(ic_data_list) > 0 and isinstance(ic_data_list[0], (list, tuple, np.ndarray)):
            ic_series = []
            for preds, actuals in ic_data_list:
                ic_val = calculate_rank_ic(np.array(preds), np.array(actuals))
                ic_series.append(ic_val)
        else:
            # Assume it's already a list of IC values
            ic_series = list(ic_data_list)

        ic_array = np.array(ic_series)
        ic_array = ic_array[~np.isnan(ic_array)]

        if len(ic_array) == 0:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'ir': np.nan, 'ic_series': ic_series}

        ic_mean = ic_array.mean()
        ic_std = ic_array.std()
        ir = ic_mean / ic_std if ic_std > 0 else np.nan

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_series': ic_series
        }

    def save(self, path):
        """Save model to file."""
        if self.booster:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            self.booster.save_model(path)
            print(f"XGBoost model saved to {path}")

            # Also save metadata
            meta_path = path + '.meta.json'
            import json
            meta = {
                'params': self.params,
                'train_history': self.train_history
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"Model metadata saved to {meta_path}")

    def load(self, path):
        """Load model from file."""
        if os.path.exists(path):
            self.booster = xgb.Booster()
            self.booster.load_model(path)
            print(f"XGBoost model loaded from {path}")

            # Load metadata if exists
            meta_path = path + '.meta.json'
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.params = meta.get('params', self.params)
                self.train_history = meta.get('train_history', [])
        else:
            print(f"File {path} not found.")
