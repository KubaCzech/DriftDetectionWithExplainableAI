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
            "n_drift_features": 2,
            "noise_percentage": 0.05,
            "drift_noise_percentage": 0.1,
            "mag_change": 0.2,
            "drift_width": 400
        })
        return params

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
            },
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
            },
            {
                "name": "noise_percentage",
                "type": "float",
                "label": "Noise Percentage (noise_percentage)",
                "default": 0.05,
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.01,
                "help": "Probability of label noise for the initial stream."
            },
            {
                "name": "drift_noise_percentage",
                "type": "float",
                "label": "Drift Noise Percentage (drift_noise_percentage)",
                "default": 0.1,
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.01,
                "help": "Probability of label noise for the drift stream."
            },
            {
                "name": "mag_change",
                "type": "float",
                "label": "Magnitude of Change (mag_change)",
                "default": 0.2,
                "min_value": 0.0,
                "step": 0.01,
                "help": "Magnitude of change for drifting features."
            },
            {
                "name": "drift_width",
                "type": "int",
                "label": "Drift Width (drift_width)",
                "default": 400,
                "min_value": 1,
                "step": 1,
                "help": "Width of the concept drift (number of samples)."
            }
        ]

    def get_available_settings(self) -> dict[str, dict]:
        return {
            "Default": {
                "n_features": 2,
                "n_drift_features": 2,
                "noise_percentage": 0.05,
                "drift_noise_percentage": 0.1,
                "mag_change": 0.2,
                "drift_width": 400
            }
        }

    # TODO: Seems that the parameter mag_change has no effect on the data stream.
    # Remove it, or fix it.
    def generate(self, n_samples_before=1000, n_samples_after=1000,
                 n_features=2, n_drift_features=2, noise_percentage=0.05,
                 drift_noise_percentage=0.1, mag_change=0.2, drift_width=400,
                 random_seed=42, **kwargs):
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
                noise_percentage=noise_percentage
            ),
            drift_stream=synth.Hyperplane(
                n_features=n_features,
                n_drift_features=n_drift_features,
                seed=random_seed,
                mag_change=mag_change,
                noise_percentage=drift_noise_percentage
            ),
            position=n_samples_before,
            width=drift_width,  # Gradual drift
            seed=random_seed
        )
        return generate_river_data(stream_HP, n_samples_before + n_samples_after, n_features)
