from .base_model import BaseModel
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr


def calculate_rank_ic(predictions, actual_returns):
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    if mask.sum() < 3:
        return np.nan
    return spearmanr(predictions[mask], actual_returns[mask]).correlation


class LightGBMWrapper(BaseModel):

    def __init__(self, name="LightGBM",
                 objective='regression',
                 num_leaves=31,
                 learning_rate=0.05,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 min_child_samples=50,
                 reg_alpha=0.5,
                 reg_lambda=1.0,
                 seed=42,
                 **kwargs):
        super().__init__(name)

        self.params = {
            'objective': objective,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_samples': min_child_samples,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'seed': seed,
            'verbosity': -1,
            'force_col_wise': True,
        }
        self.params.update(kwargs)

        self.booster = None
        self.train_history = []
        self.feature_names = None

    def _prepare_data(self, X):
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['category']).columns:
            X_clean[col] = X_clean[col].cat.codes.astype(np.int32)
        X_clean = X_clean.select_dtypes(include=['number']).fillna(0)
        return X_clean

    def _get_eval_metric(self):
        objective = self.params.get('objective', 'regression')
        if 'lambdarank' in objective or 'rank' in objective:
            return 'ndcg'
        elif 'binary' in objective:
            return 'auc'
        elif 'multiclass' in objective:
            return 'multi_logloss'
        else:
            return 'rmse'

    def train(self, X, y, num_boost_round=100, groups=None, eval_X=None, eval_y=None,
              early_stopping_rounds=10, eval_ic=False):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        X_clean = self._prepare_data(X)
        self.feature_names = list(X_clean.columns)
        dtrain = lgb.Dataset(X_clean, label=y, group=groups)

        valid_sets = [dtrain]
        valid_names = ['train']

        if eval_X is not None and eval_y is not None:
            X_val_clean = self._prepare_data(eval_X)
            dval = lgb.Dataset(X_val_clean, label=eval_y, group=groups)
            valid_sets.append(dval)
            valid_names.append('eval')

        use_early_stop = (eval_X is not None and early_stopping_rounds > 0)

        print(f"[{self.name}] Training with {num_boost_round} rounds, objective: {self.params.get('objective')}")

        eval_metric = self._get_eval_metric()
        train_params = {**self.params, 'metric': eval_metric}

        callbacks = []
        if use_early_stop:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=True))
            callbacks.append(lgb.log_evaluation(10))

        self.booster = lgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks if callbacks else [lgb.log_evaluation(10)]
        )

        best_iteration = self.booster.current_iteration()
        print(f"[{self.name}] Training complete. Best iteration: {best_iteration}")

        if eval_ic:
            train_preds = self.booster.predict(X_clean)
            ic = calculate_rank_ic(train_preds, y)
            print(f"[{self.name}] Rank IC: {ic:.4f}")
            self.train_history.append({'type': 'full', 'rounds': best_iteration, 'rank_ic': ic})
        else:
            self.train_history.append({'type': 'full', 'rounds': best_iteration})

        return self.booster

    def partial_train(self, X, y, num_boost_round=10, groups=None):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        X_clean = self._prepare_data(X)
        dtrain = lgb.Dataset(X_clean, label=y, group=groups)

        if self.booster is None:
            print(f"[{self.name}] No existing model. Starting full train.")
            self.train(X, y, num_boost_round=num_boost_round, groups=groups)
            return

        incremental_params = {**self.params}
        incremental_params['learning_rate'] = self.params.get('learning_rate', 0.05) * 0.5

        print(f"[{self.name}] Incremental training for {num_boost_round} rounds...")
        self.booster = lgb.train(
            incremental_params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.booster,
            callbacks=[lgb.log_evaluation(5)]
        )
        print(f"[{self.name}] Incremental training complete.")
        self.train_history.append({'type': 'incremental', 'rounds': num_boost_round})

    def predict(self, X):
        if self.booster is None:
            raise ValueError("Model is not trained yet.")
        X_clean = self._prepare_data(X)
        if self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X_clean.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=X_clean.index, columns=missing_cols)
                X_clean = pd.concat([X_clean, missing_df], axis=1)
            X_clean = X_clean[self.feature_names]
        return self.booster.predict(X_clean)

    def get_feature_importance(self, importance_type='gain'):
        if self.booster is None:
            raise ValueError("Model is not trained yet.")
        names = self.booster.feature_name()
        gains = self.booster.feature_importance(importance_type=importance_type)
        return dict(zip(names, gains))

    def save(self, path):
        if self.booster:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            self.booster.save_model(path)
            print(f"[{self.name}] Model saved to {path}")
            _save_meta(path, self.params, self.train_history, self.feature_names)

    def load(self, path):
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        if os.path.exists(path):
            self.booster = lgb.Booster(model_file=path)
            print(f"[{self.name}] Model loaded from {path}")
            meta = _load_meta(path)
            if meta:
                self.params = meta.get('params', self.params)
                self.train_history = meta.get('train_history', [])
                self.feature_names = meta.get('feature_names', None)
        else:
            print(f"File {path} not found.")


def _save_meta(model_path, params, history, feature_names):
    meta_path = model_path + '.meta.json'
    import json
    meta = {
        'params': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v for k, v in params.items()},
        'train_history': history,
        'feature_names': feature_names
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def _load_meta(model_path):
    meta_path = model_path + '.meta.json'
    import json
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None
