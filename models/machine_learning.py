from .base_model import BaseModel
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import os

class SklearnWrapper(BaseModel):
    """
    Wrapper for Scikit-learn models supporting:
    - Incremental learning (partial_fit) when available
    - Training evaluation metrics
    - Model save/load with metadata

    Supports both regression and classification objectives.
    """

    # 默认模型映射，根据 objective 类型选择
    DEFAULT_MODELS = {
        'regressor': SGDRegressor(random_state=42),
        'classifier': SGDClassifier(random_state=42)
    }

    def __init__(self, model=None, model_type='auto', name="SklearnModel"):
        """
        Args:
            model: Optional sklearn estimator. If None, uses default based on model_type.
            model_type: 'auto', 'regressor', or 'classifier'. Auto-detects from model if possible.
            name: Model name for logging.
        """
        super().__init__(name)
        self.model = model
        self.model_type = model_type
        self.train_history = []

        # Auto-detect model type if not specified
        if model is not None:
            if isinstance(model, ClassifierMixin):
                self.model_type = 'classifier'
            elif isinstance(model, RegressorMixin):
                self.model_type = 'regressor'
        elif model_type == 'auto':
            self.model_type = 'regressor'  # Default

        # Set default model if not provided
        if self.model is None:
            if self.model_type == 'classifier':
                self.model = SGDClassifier(random_state=42)
            else:
                self.model = SGDRegressor(random_state=42)

    def _prepare_data(self, X):
        """Prepare features: select numeric types and fill NaN."""
        X_clean = X.select_dtypes(include=['number']).fillna(0)
        return X_clean

    def _get_eval_score(self, X, y):
        """Calculate evaluation score based on model type."""
        try:
            if self.model_type == 'classifier':
                from sklearn.metrics import accuracy_score
                preds = self.model.predict(X)
                return accuracy_score(y, preds)
            else:
                from sklearn.metrics import r2_score
                preds = self.model.predict(X)
                return r2_score(y, preds)
        except Exception:
            return None

    def train(self, X, y, eval_X=None, eval_y=None):
        """
        Train the sklearn model.

        Args:
            X: Training features
            y: Training labels
            eval_X: Optional validation features
            eval_y: Optional validation labels

        Returns:
            dict: Training info including evaluation score
        """
        X_clean = self._prepare_data(X)
        self.model.fit(X_clean, y)

        # Calculate training score
        train_score = self._get_eval_score(X_clean, y)

        print(f"[{self.name}] Training complete.", end="")
        if train_score is not None:
            metric_name = "Accuracy" if self.model_type == 'classifier' else "R2"
            print(f" Train {metric_name}: {train_score:.4f}")
        else:
            print()

        # Calculate validation score if provided
        if eval_X is not None and eval_y is not None:
            X_val_clean = self._prepare_data(eval_X)
            val_score = self._get_eval_score(X_val_clean, eval_y)
            if val_score is not None:
                metric_name = "Accuracy" if self.model_type == 'classifier' else "R2"
                print(f"[{self.name}] Validation {metric_name}: {val_score:.4f}")

        self.train_history.append({
            'type': 'full',
            'train_score': train_score
        })

    def partial_train(self, X, y):
        """
        Incrementally update the model using new data.

        Falls back to full retrain if model doesn't support partial_fit.

        Args:
            X: New training features
            y: New training labels
        """
        X_clean = self._prepare_data(X)

        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_clean, y)
            print(f"[{self.name}] Incremental training complete (partial_fit).")
            self.train_history.append({'type': 'incremental', 'method': 'partial_fit'})
        else:
            print(f"[{self.name}] Model does not support partial_fit. Falling back to full retrain.")
            self.train(X, y)
            self.train_history.append({'type': 'incremental', 'method': 'full_retrain'})

    def predict(self, X):
        """Predict using trained model."""
        X_clean = self._prepare_data(X)
        return self.model.predict(X_clean)

    def predict_proba(self, X):
        """
        Predict probability (for classifiers only).

        Returns:
            array of shape (n_samples, n_classes) or (n_samples, 2) for binary
        """
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {type(self.model)} does not support predict_proba")
        X_clean = self._prepare_data(X)
        return self.model.predict_proba(X_clean)

    def get_feature_importance(self):
        """
        Get feature importance if available.

        Returns:
            dict: Feature name -> importance score, or None if not supported
        """
        # RandomForest has feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            if hasattr(self.model, 'feature_names_in_'):
                return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
            else:
                return dict(enumerate(self.model.feature_importances_))

        # Linear models have coef_
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_.flatten()
            return dict(enumerate(coef))

        return None

    def save(self, path):
        """Save model to file with metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

        # Save metadata
        meta_path = path + '.meta.json'
        import json
        meta = {
            'model_type': self.model_type,
            'model_class': type(self.model).__name__,
            'train_history': self.train_history
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Model metadata saved to {meta_path}")

    def load(self, path):
        """Load model from file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {path}")

            # Update model type based on loaded model
            if isinstance(self.model, ClassifierMixin):
                self.model_type = 'classifier'
            elif isinstance(self.model, RegressorMixin):
                self.model_type = 'regressor'

            # Load metadata if exists
            meta_path = path + '.meta.json'
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.train_history = meta.get('train_history', [])
        else:
            print(f"File {path} not found.")
