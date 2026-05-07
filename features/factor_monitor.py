import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import deque


class FactorICMonitor:

    def __init__(self, max_history=60, stale_threshold=0.02, stale_days=20):
        self.max_history = max_history
        self.stale_threshold = stale_threshold
        self.stale_days = stale_days

        self.ic_history = {}
        self.factor_status = {}

    def update(self, date, factor_df, forward_returns):
        numeric_cols = factor_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col.startswith('target_'):
                continue
            values = factor_df[col]
            mask = ~(values.isna() | forward_returns.isna())
            if mask.sum() < 10:
                continue

            ic = spearmanr(values[mask], forward_returns[mask]).correlation

            if col not in self.ic_history:
                self.ic_history[col] = deque(maxlen=self.max_history)

            self.ic_history[col].append({'date': date, 'ic': ic if not np.isnan(ic) else 0.0})

        self._update_status()

    def _update_status(self):
        for factor, history in self.ic_history.items():
            ics = [h['ic'] for h in history]
            recent_ics = ics[-self.stale_days:] if len(ics) >= self.stale_days else ics

            mean_ic = np.mean(recent_ics) if recent_ics else 0
            ic_std = np.std(recent_ics) if len(recent_ics) > 1 else 0
            icir = mean_ic / ic_std if ic_std > 0 else 0

            abs_mean_ic = abs(mean_ic)

            if abs_mean_ic < self.stale_threshold and len(recent_ics) >= self.stale_days:
                status = 'stale'
            elif len(recent_ics) < 5:
                status = 'insufficient_data'
            elif abs_mean_ic < 0.03:
                status = 'weak'
            elif abs_mean_ic < 0.05:
                status = 'moderate'
            else:
                status = 'strong'

            self.factor_status[factor] = {
                'mean_ic': mean_ic,
                'ic_std': ic_std,
                'icir': icir,
                'status': status,
                'n_obs': len(recent_ics)
            }

    def get_stale_factors(self):
        return [f for f, s in self.factor_status.items() if s['status'] == 'stale']

    def get_strong_factors(self):
        return [f for f, s in self.factor_status.items() if s['status'] == 'strong']

    def get_factor_weights(self, decay_factor=0.95):
        weights = {}
        for factor, status in self.factor_status.items():
            if status['status'] == 'stale':
                weights[factor] = 0.0
            elif status['status'] == 'insufficient_data':
                weights[factor] = 0.5
            elif status['icir'] > 0:
                weights[factor] = status['icir']
            else:
                weights[factor] = max(0, status['mean_ic'])

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def summary(self):
        lines = []
        sorted_factors = sorted(self.factor_status.items(),
                                key=lambda x: abs(x[1].get('mean_ic', 0)), reverse=True)
        for factor, status in sorted_factors:
            lines.append(
                f"  {factor:<40} "
                f"IC={status['mean_ic']:+.4f} "
                f"ICIR={status['icir']:+.3f} "
                f"[{status['status']}]"
            )
        return "\n".join(lines)
