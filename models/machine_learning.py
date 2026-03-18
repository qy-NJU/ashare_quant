from .base_model import BaseModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDRegressor
import pandas as pd
import pickle

class SklearnWrapper(BaseModel):
    """
    Wrapper for Scikit-learn models supporting incremental learning (partial_fit).
    """
    def __init__(self, model: BaseEstimator = None, name="SklearnModel"):
        super().__init__(name)
        # Default to SGDRegressor which supports partial_fit
        self.model = model if model else SGDRegressor(random_state=42)

    def train(self, X, y):
        X_clean = X.select_dtypes(include=['number']).fillna(0)
        self.model.fit(X_clean, y)
        print(f"[{self.name}] Full training complete. Score: {self.model.score(X_clean, y):.4f}")

    def partial_train(self, X, y):
        X_clean = X.select_dtypes(include=['number']).fillna(0)
        
        # Check if model supports partial_fit
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_clean, y)
            print(f"[{self.name}] Incremental training complete.")
        else:
            print(f"[{self.name}] Model does not support partial_fit. Falling back to full retrain on new batch.")
            self.model.fit(X_clean, y)

    def predict(self, X):
        X_clean = X.select_dtypes(include=['number']).fillna(0)
        return self.model.predict(X_clean)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")
