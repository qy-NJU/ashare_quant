import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelEvaluator:
    """
    Evaluates the predictive performance of a machine learning model,
    independent of trading strategy rules.
    """
    
    @staticmethod
    def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Evaluate regression model predictions.
        
        Args:
            y_true: Actual returns (or targets)
            y_pred: Predicted returns (or targets)
            
        Returns:
            dict containing regression metrics
        """
        # Ensure we have aligned, non-NaN data
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
        if len(df) == 0:
            return {}
            
        true_vals = df['y_true'].values
        pred_vals = df['y_pred'].values
        
        # Calculate MSE and MAE
        mse = mean_squared_error(true_vals, pred_vals)
        mae = mean_absolute_error(true_vals, pred_vals)
        
        # Calculate IC (Pearson correlation) and Rank IC (Spearman correlation)
        ic, _ = pearsonr(true_vals, pred_vals)
        rank_ic, _ = spearmanr(true_vals, pred_vals)
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'IC': float(ic),
            'Rank_IC': float(rank_ic)
        }

    @staticmethod
    def evaluate_classification(y_true: pd.Series, y_pred_prob: pd.Series, threshold: float = 0.5) -> dict:
        """
        Evaluate classification model predictions.
        
        Args:
            y_true: Actual binary labels (e.g., 1 for positive return, 0 for negative)
            y_pred_prob: Predicted probabilities of the positive class
            threshold: Probability threshold for classification
            
        Returns:
            dict containing classification metrics
        """
        df = pd.DataFrame({'y_true': y_true, 'y_pred_prob': y_pred_prob}).dropna()
        if len(df) == 0:
            return {}
            
        true_vals = df['y_true'].values
        prob_vals = df['y_pred_prob'].values
        pred_classes = (prob_vals >= threshold).astype(int)
        
        metrics = {
            'Accuracy': float(accuracy_score(true_vals, pred_classes)),
            'Precision': float(precision_score(true_vals, pred_classes, zero_division=0)),
            'Recall': float(recall_score(true_vals, pred_classes, zero_division=0)),
            'F1_Score': float(f1_score(true_vals, pred_classes, zero_division=0))
        }
        
        try:
            # AUC requires both classes to be present in y_true
            if len(np.unique(true_vals)) > 1:
                metrics['AUC'] = float(roc_auc_score(true_vals, prob_vals))
            else:
                metrics['AUC'] = np.nan
        except ValueError:
            metrics['AUC'] = np.nan
            
        return metrics
        
    @staticmethod
    def calculate_ic_series(df: pd.DataFrame, date_col: str, target_col: str, pred_col: str) -> pd.DataFrame:
        """
        Calculate IC and Rank IC grouped by date (cross-sectional IC).
        
        Args:
            df: DataFrame containing date, target, and prediction columns
            date_col: Column name for dates
            target_col: Column name for actual targets
            pred_col: Column name for predictions
            
        Returns:
            DataFrame with IC and Rank IC for each date
        """
        def calc_corrs(group):
            if len(group) < 2:
                return pd.Series({'IC': np.nan, 'Rank_IC': np.nan})
            
            t = group[target_col].values
            p = group[pred_col].values
            
            # Add small noise to handle constant predictions which cause pearsonr/spearmanr to return NaN
            if np.std(p) < 1e-8:
                p = p + np.random.normal(0, 1e-8, len(p))
            if np.std(t) < 1e-8:
                t = t + np.random.normal(0, 1e-8, len(t))
                
            ic, _ = pearsonr(t, p)
            rank_ic, _ = spearmanr(t, p)
            
            return pd.Series({'IC': ic, 'Rank_IC': rank_ic})
            
        ic_series = df.groupby(date_col).apply(calc_corrs).reset_index()
        return ic_series
