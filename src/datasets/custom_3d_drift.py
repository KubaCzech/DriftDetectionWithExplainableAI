import numpy as np
from .base import BaseDataset

class Custom3DDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "custom_3d_drift"

    @property
    def display_name(self) -> str:
        return "Custom 3D Drift"

    def generate(self, n_samples_before=1000, n_samples_after=1000, random_seed=42, **kwargs):
        """
        Generate synthetic 3D data stream with concept drift.
        """
        np.random.seed(random_seed)

        # Before drift
        X_before = np.random.normal(0, 1, (n_samples_before, 3))
        scores_before = (X_before[:, 0] + X_before[:, 1] + X_before[:, 2] +
                         np.random.normal(0, 0.5, n_samples_before))
        threshold_before = np.percentile(scores_before, 40)
        y_before = (scores_before > threshold_before).astype(int)

        # After drift
        X1_after = np.random.normal(0, 1, n_samples_after)
        X2_after = np.random.normal(0, 1, n_samples_after)
        X3_after = np.random.normal(3, 1.5, n_samples_after)  # Data drift on X3
        X_after = np.column_stack([X1_after, X2_after, X3_after])

        # Concept drift: X3 becomes more important and inverts relationship
        scores_after = (X_after[:, 0] + X_after[:, 1] - 2 * X_after[:, 2] +
                        np.random.normal(0, 0.5, n_samples_after))
        threshold_after = np.percentile(scores_after, 60)
        y_after = (scores_after > threshold_after).astype(int)

        # Combine data
        X = np.concatenate([X_before, X_after])
        y = np.concatenate([y_before, y_after])
        drift_point = n_samples_before
        feature_names = ['X1', 'X2', 'X3']

        return X, y, drift_point, feature_names
