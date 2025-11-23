import numpy as np
from .base import BaseDataset

class ControlledConceptDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "controlled_concept_drift"

    @property
    def display_name(self) -> str:
        return "Controlled Concept Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "n_features": 4,
            "n_drift_features": 2
        })
        return params

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_features",
                "type": "int",
                "label": "Number of Features (n_features)",
                "default": 11,
                "min_value": 2,
                "step": 1,
                "help": "Total number of features for the dataset. Must be >= 2."
            },
            {
                "name": "n_drift_features",
                "type": "int",
                "label": "Number of Drifting Features (n_drift_features)",
                "default": 5,
                "min_value": 1,
                "step": 1,
                "help": "Number of features that will drift. Must be <= n_features."
            }
        ]

    def generate(self, n_samples_before=1000, n_samples_after=1000,
                 n_features=4, n_drift_features=2, random_seed=42, **kwargs):
        """
        Generate synthetic data with controlled Concept Drift.
        """
        # Validation
        if n_drift_features > n_features:
            raise ValueError("n_drift_features cannot exceed n_features")
        if n_drift_features == 0:
            raise ValueError("n_drift_features must be greater than 0 "
                             "to observe drift")

        np.random.seed(random_seed)
        total_samples = n_samples_before + n_samples_after
        drift_point = n_samples_before
        feature_names = [f'X{i+1}' for i in range(n_features)]

        # --- Feature Generation (NO Data Drift) ---
        # Generate all features from a stable distribution (Uniform [0, 1])
        X = np.random.uniform(0, 1, (total_samples, n_features))

        # --- Define Weights for Equal Magnitude Concept Drift ---

        # Base weight for non-drifting features (e.g., X3, X4, ...)
        stable_weight = 0.1
        stable_weights = np.full(n_features - n_drift_features, stable_weight)

        # Weights for the drifting features (e.g., X1, X2)
        # 1. Before Drift: All drifting features have a strong positive influence.
        drift_weight_before = 1.0
        weights_drift_before = np.full(n_drift_features, drift_weight_before)
        W_before = np.concatenate([weights_drift_before, stable_weights])

        # 2. After Drift: All drifting features have an equally strong
        # negative influence.
        drift_weight_after = -1.0
        weights_drift_after = np.full(n_drift_features, drift_weight_after)
        W_after = np.concatenate([weights_drift_after, stable_weights])

        # --- Calculate Scores and Labels ---

        # Calculate scores before drift (0 to drift_point)
        scores_before = (X[:drift_point] @ W_before +
                         np.random.normal(0, 0.1, n_samples_before))
        threshold_before = np.percentile(scores_before, 50)
        y_before = (scores_before > threshold_before).astype(int)

        # Calculate scores after drift (drift_point to end)
        scores_after = (X[drift_point:] @ W_after +
                        np.random.normal(0, 0.1, n_samples_after))
        threshold_after = np.percentile(scores_after, 50)
        y_after = (scores_after > threshold_after).astype(int)

        # Combine data
        y = np.concatenate([y_before, y_after])

        return X, y, drift_point, feature_names
