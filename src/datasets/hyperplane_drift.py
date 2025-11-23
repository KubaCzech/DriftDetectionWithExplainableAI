from river.datasets import synth
from .base import BaseDataset
from .utils import generate_river_data

class HyperplaneDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "hyperplane_drift"

    @property
    def display_name(self) -> str:
        return "Hyperplane Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "n_features": 2,
            "n_drift_features": 2
        })
        return params

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_features",
                "type": "int",
                "label": "Number of Features (n_features)",
                "default": 2,
                "min_value": 2,
                "step": 1,
                "help": "Total number of features for the hyperplane. Must be >= 2."
            },
            {
                "name": "n_drift_features",
                "type": "int",
                "label": "Number of Drifting Features (n_drift_features)",
                "default": 2,
                "min_value": 2,
                "step": 1,
                "help": "Number of features that will drift. Must be <= n_features."
            }
        ]

    def generate(self, n_samples_before=1000, n_samples_after=1000,
                 n_features=2, n_drift_features=2, random_seed=42, **kwargs):
        """
        Generate synthetic data stream using River's Hyperplane generator.
        """
        # Validation
        if n_features < 2:
            raise ValueError("n_features must be at least 2")
        if n_drift_features < 2:
            raise ValueError("n_drift_features must be at least 2")
        if n_drift_features > n_features:
            raise ValueError("n_drift_features cannot exceed n_features")

        stream_HP = synth.ConceptDriftStream(
            stream=synth.Hyperplane(
                n_features=n_features,
                n_drift_features=n_drift_features,
                seed=random_seed,
                noise_percentage=0.05
            ),
            drift_stream=synth.Hyperplane(
                n_features=n_features,
                n_drift_features=n_drift_features,
                seed=random_seed,
                mag_change=0.2,
                noise_percentage=0.1
            ),
            position=n_samples_before,
            width=400,  # Gradual drift
            seed=random_seed
        )
        return generate_river_data(stream_HP, n_samples_before,
                                    n_samples_after, n_features)
