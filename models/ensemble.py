import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import spearmanr


class ModelEnsemble:

    def __init__(self, models=None, market_state_recognizer=None, name="ModelEnsemble"):
        self.models = models if models is not None else []
        self.market_state_recognizer = market_state_recognizer
        self.name = name

        self.ic_history = {}
        self.weight_history = []
        self.state_weights = {}

    def add_model(self, model):
        self.models.append(model)

    def train_all(self, X, y, groups=None, eval_X=None, eval_y=None, **train_kwargs):
        for model in self.models:
            print(f"\n{'='*50}")
            print(f"[{self.name}] Training {model.name}...")
            print(f"{'='*50}")
            try:
                model.train(X, y, groups=groups, eval_X=eval_X, eval_y=eval_y, **train_kwargs)
            except Exception as e:
                print(f"[{self.name}] Failed to train {model.name}: {e}")

    def update_ic_tracking(self, predictions_by_model, y_true, date=None):
        for model_name, preds in predictions_by_model.items():
            mask = ~(np.isnan(preds) | np.isnan(y_true))
            if mask.sum() < 3:
                continue
            ic = spearmanr(preds[mask], y_true[mask]).correlation

            if model_name not in self.ic_history:
                self.ic_history[model_name] = []
            self.ic_history[model_name].append({
                'date': date,
                'ic': ic
            })

    def _get_model_recent_ic(self, model_name, lookback=20):
        if model_name not in self.ic_history or not self.ic_history[model_name]:
            return 0.0
        recent = self.ic_history[model_name][-lookback:]
        ics = [entry['ic'] for entry in recent if not np.isnan(entry['ic'])]
        return np.mean(ics) if ics else 0.0

    def _get_model_ic_stability(self, model_name, lookback=20):
        if model_name not in self.ic_history or not self.ic_history[model_name]:
            return 1.0
        recent = self.ic_history[model_name][-lookback:]
        ics = [entry['ic'] for entry in recent if not np.isnan(entry['ic'])]
        if len(ics) < 2:
            return 1.0
        std = np.std(ics)
        return 1.0 / (1.0 + std)

    def compute_weights(self, market_state=None, temperature=1.0):
        weights = {}
        model_names = [m.name for m in self.models]

        if market_state and market_state in self.state_weights:
            return self.state_weights[market_state].copy()

        if not self.ic_history:
            w = 1.0 / len(self.models)
            weights = {m.name: w for m in self.models}
            self.weight_history.append({'state': market_state, 'weights': weights})
            return weights

        scores = {}
        for name in model_names:
            mean_ic = self._get_model_recent_ic(name)
            stability = self._get_model_ic_stability(name)
            scores[name] = mean_ic * stability

        min_score = min(scores.values()) if scores else -1
        if min_score < 0:
            shift = abs(min_score) + 0.01
            scores = {k: v + shift for k, v in scores.items()}

        total = sum(scores.values())
        if total > 0:
            raw_weights = {k: v / total for k, v in scores.items()}
        else:
            w = 1.0 / len(model_names)
            raw_weights = {k: w for k in model_names}

        if temperature != 1.0:
            raw_weights = {k: v ** (1.0 / temperature) for k, v in raw_weights.items()}
            total = sum(raw_weights.values())
            if total > 0:
                raw_weights = {k: v / total for k, v in raw_weights.items()}

        weights = raw_weights
        self.weight_history.append({
            'state': market_state,
            'scores': scores,
            'weights': weights
        })
        return weights

    def predict(self, X, market_state=None, return_raw=False):
        if not self.models:
            raise ValueError("No models in ensemble.")

        predictions = {}
        for model in self.models:
            try:
                preds = model.predict(X)
                predictions[model.name] = preds
            except Exception as e:
                print(f"[{self.name}] Model {model.name} prediction failed: {e}")
                predictions[model.name] = np.zeros(len(X))

        weights = self.compute_weights(market_state=market_state)

        combined = np.zeros(len(X))
        for name, preds in predictions.items():
            w = weights.get(name, 0.0)
            if w > 0:
                combined += w * preds

        if return_raw:
            return combined

        return combined, predictions, weights

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        for model in self.models:
            model_path = os.path.join(save_dir, f"{model.name}_model")
            model.save(model_path)

        ensemble_meta = {
            'model_names': [m.name for m in self.models],
            'ic_history': self.ic_history,
            'weight_history': self.weight_history,
            'state_weights': self.state_weights
        }
        meta_path = os.path.join(save_dir, 'ensemble_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(ensemble_meta, f)
        print(f"[{self.name}] Ensemble metadata saved to {meta_path}")

    def load(self, save_dir, model_builders):
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Ensemble directory {save_dir} not found.")

        meta_path = os.path.join(save_dir, 'ensemble_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.ic_history = meta.get('ic_history', {})
            self.weight_history = meta.get('weight_history', [])
            self.state_weights = meta.get('state_weights', {})

        self.models = []
        for model_builder in model_builders:
            name = model_builder.__class__.__name__
            model_path = os.path.join(save_dir, f"{name}_model")
            if os.path.exists(model_path):
                model_builder.load(model_path)
                self.models.append(model_builder)

        print(f"[{self.name}] Loaded {len(self.models)} models from {save_dir}")
