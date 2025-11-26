import numpy as np
import pandas as pd
from .base import BaseDataset


class CustomNormalDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "custom_normal"

    @property
    def display_name(self) -> str:
        return "Custom Normal"

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_samples_before",
                "type": "int",
                "label": "Number of Samples Before Drift",
                "default": 1000,
                "min_value": 100,
                "step": 100,
                "help": "Number of samples generated before the concept drift occurs."
            },
            {
                "name": "n_samples_after",
                "type": "int",
                "label": "Number of Samples After Drift",
                "default": 1000,
                "min_value": 100,
                "step": 100,
                "help": "Number of samples generated after the concept drift occurs."
            }
        ]

    def generate(self, n_samples_before=1000, n_samples_after=1000, random_seed=42, **kwargs):
        """
        Generate synthetic data stream with concept drift (Original Function, 2D).
        """
        np.random.seed(random_seed)

        # Before drift
        X1_before = np.random.normal(0, 1, n_samples_before)
        X2_before = np.random.normal(0, 1, n_samples_before)
        scores_before = (2 * X1_before + 0.5 * X2_before +
                         np.random.normal(0, 0.5, n_samples_before))
        threshold_before = np.percentile(scores_before, 30)
        y_before = (scores_before > threshold_before).astype(int)

        # After drift - X1 distribution changes DRAMATICALLY, X2 changes slightly
        X1_after = np.random.normal(2, 1.5, n_samples_after)
        X2_after = np.random.normal(0.2, 1.05, n_samples_after)
        scores_after = (-1.5 * X1_after + 0.6 * X2_after +
                        np.random.normal(0, 0.5, n_samples_after))
        threshold_after = np.percentile(scores_after, 70)
        y_after = (scores_after > threshold_after).astype(int)

        # Combine data
        # Combine data
        X1 = np.concatenate([X1_before, X1_after])
        X2 = np.concatenate([X2_before, X2_after])
        X = np.column_stack([X1, X2])
        y = np.concatenate([y_before, y_after])
        drift_point = n_samples_before
        feature_names = ['X1', 'X2']

        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='Y')

        return X_df, y_series, drift_point
