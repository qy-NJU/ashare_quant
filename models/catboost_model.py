from .base_model import BaseModel
import pandas as pd
import numpy as np
import os


class CatBoostWrapper(BaseModel):

    def __init__(self, name="CatBoost",
                 loss_function='RMSE',
                 depth=6,
                 learning_rate=0.05,
                 subsample=0.8,
                 l2_leaf_reg=3.0,
                 random_seed=42,
                 **kwargs):
        super().__init__(name)

        self.params = {
            'loss_function': loss_function,
            'depth': depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'l2_leaf_reg': l2_leaf_reg,
            'random_seed': random_seed,
            'verbose': False,
            'allow_writing_files': False,
        }
        self.params.update(kwargs)

        self.model = None
        self.train_history = []
        self.feature_names = None

    def _prepare_data(self, X):
        X_clean = X.copy()
        for col in X_clean.select_dtypes(include=['category']).columns:
            X_clean[col] = X_clean[col].cat.codes.astype(np.float64)
        X_clean = X_clean.select_dtypes(include=['number']).fillna(0)
        return X_clean

    def _get_cat_features(self, X):
        cat_cols = X.select_dtypes(include=['category']).columns.tolist()
        cat_indices = [i for i, col in enumerate(X.columns) if col in cat_cols]
        return cat_indices if cat_indices else None

    def train(self, X, y, num_boost_round=100, groups=None, eval_X=None, eval_y=None,
              early_stopping_rounds=10, eval_ic=False):
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier, CatBoost, Pool
        except ImportError:
            raise ImportError("catboost is not installed. Run: pip install catboost")

        X_clean = self._prepare_data(X)
        self.feature_names = list(X_clean.columns)

        loss = self.params.get('loss_function', 'RMSE')
        is_classifier = loss in ('Logloss', 'CrossEntropy', 'MultiClass')

        if is_classifier:
            self.model = CatBoostClassifier(**self.params, iterations=num_boost_round)
        else:
            self.model = CatBoostRegressor(**self.params, iterations=num_boost_round)

        cat_features = self._get_cat_features(X_clean)

        eval_set = None
        if eval_X is not None and eval_y is not None:
            X_val_clean = self._prepare_data(eval_X)
            eval_set = (X_val_clean, eval_y)

        if groups is not None and eval_set is None:
            eval_set = (X_clean, y)

        use_early_stop = eval_set is not None and early_stopping_rounds > 0

        print(f"[{self.name}] Training with {num_boost_round} rounds, loss: {loss}")

        self.model.fit(
            X_clean, y,
            cat_features=cat_features,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if use_early_stop else None,
            verbose=10
        )

        best_iteration = self.model.get_best_iteration() if self.model.get_best_iteration() else num_boost_round
        print(f"[{self.name}] Training complete. Best iteration: {best_iteration}")

        if eval_ic:
            train_preds = self.model.predict(X_clean)
            from scipy.stats import spearmanr
            mask = ~(np.isnan(train_preds) | np.isnan(y))
            if mask.sum() >= 3:
                ic = spearmanr(train_preds[mask], y[mask]).correlation
                print(f"[{self.name}] Rank IC: {ic:.4f}")
                self.train_history.append({'type': 'full', 'rounds': best_iteration, 'rank_ic': ic})
        else:
            self.train_history.append({'type': 'full', 'rounds': best_iteration})

    def partial_train(self, X, y, num_boost_round=10, groups=None):
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            raise ImportError("catboost is not installed. Run: pip install catboost")

        X_clean = self._prepare_data(X)
        cat_features = self._get_cat_features(X_clean)

        if self.model is None:
            print(f"[{self.name}] No existing model. Starting full train.")
            self.train(X, y, num_boost_round=num_boost_round, groups=groups)
            return

        self.model.fit(
            X_clean, y,
            cat_features=cat_features,
            init_model=self.model,
            verbose=5
        )
        print(f"[{self.name}] Incremental training complete.")
        self.train_history.append({'type': 'incremental', 'rounds': num_boost_round})

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        X_clean = self._prepare_data(X)
        if self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X_clean.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=X_clean.index, columns=missing_cols)
                X_clean = pd.concat([X_clean, missing_df], axis=1)
            X_clean = X_clean[self.feature_names]
        preds = self.model.predict(X_clean)
        return np.array(preds).flatten()

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        importances = self.model.get_feature_importance()
        names = self.model.feature_names_ if hasattr(self.model, 'feature_names_') else self.feature_names
        if names is None:
            names = [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances))

    def save(self, path):
        if self.model:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            self.model.save_model(path)
            print(f"[{self.name}] Model saved to {path}")
            _save_meta(path, self.params, self.train_history, self.feature_names)

    def load(self, path):
        try:
            from catboost import CatBoostRegressor, CatBoostClassifier
        except ImportError:
            raise ImportError("catboost is not installed. Run: pip install catboost")

        if os.path.exists(path):
            loss = self.params.get('loss_function', 'RMSE')
            is_classifier = loss in ('Logloss', 'CrossEntropy', 'MultiClass')
            if is_classifier:
                self.model = CatBoostClassifier()
            else:
                self.model = CatBoostRegressor()
            self.model.load_model(path)
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
