from .base_model import BaseModel
import xgboost as xgb
import pandas as pd
import os

class XGBoostWrapper(BaseModel):
    """
    Wrapper for XGBoost supporting incremental learning via xgb_model parameter.
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

    def train(self, X, y, num_boost_round=50, groups=None):
        # Allow numeric AND category types
        X_clean = X.select_dtypes(include=['number', 'category']).fillna(0)
        # Enable categorical support natively
        dtrain = xgb.DMatrix(X_clean, label=y, enable_categorical=True)
        
        if groups is not None:
            dtrain.set_group(groups)
            
        self.booster = xgb.train(self.params, dtrain, num_boost_round=num_boost_round)
        print(f"[{self.name}] Full training complete.")

    def partial_train(self, X, y, num_boost_round=10, groups=None):
        """
        Incrementally update the model using new data.
        """
        X_clean = X.select_dtypes(include=['number', 'category']).fillna(0)
        dtrain = xgb.DMatrix(X_clean, label=y, enable_categorical=True)
        
        if groups is not None:
            dtrain.set_group(groups)
            
        if self.booster is None:
            print(f"[{self.name}] No existing model found. Starting full train.")
            self.train(X, y, num_boost_round=num_boost_round, groups=groups)
        else:
            # Continue training from existing booster
            self.booster = xgb.train(
                self.params, 
                dtrain, 
                num_boost_round=num_boost_round, 
                xgb_model=self.booster
            )
            print(f"[{self.name}] Incremental training complete.")

    def predict(self, X):
        if self.booster is None:
            raise ValueError("Model is not trained yet.")
        X_clean = X.select_dtypes(include=['number', 'category']).fillna(0)
        dtest = xgb.DMatrix(X_clean, enable_categorical=True)
        return self.booster.predict(dtest)

    def save(self, path):
        if self.booster:
            self.booster.save_model(path)
            print(f"XGBoost model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.booster = xgb.Booster()
            self.booster.load_model(path)
            print(f"XGBoost model loaded from {path}")
        else:
            print(f"File {path} not found.")
