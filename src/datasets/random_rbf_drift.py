from river.datasets import synth
from .base import BaseDataset
from .utils import generate_river_data


class RandomRBFDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "random_rbf_drift"

    @property
    def display_name(self) -> str:
        return "Random RBF Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "n_features": 10,
            "n_classes": 2,
            "n_centroids": 50,
            "change_speed": 0.0,
            "n_drift_centroids": 50,
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
                "label": "Number of Features",
                "default": 10,
                "min_value": 2,
                "step": 1,
                "help": "Number of numerical features."
            },
            # {
            #     "name": "n_classes",
            #     "type": "int",
            #     "label": "Number of Classes",
            #     "default": 2,
            #     "min_value": 2,
            #     "step": 1,
            #     "help": "Number of class labels."
            # },
            {
                "name": "n_centroids",
                "type": "int",
                "label": "Number of Centroids",
                "default": 50,
                "min_value": 1,
                "step": 1,
                "help": "Total number of centroids."
            },
            {
                "name": "change_speed",
                "type": "float",
                "label": "Change Speed",
                "default": 0.0,
                "min_value": 0.0,
                "step": 0.01,
                "help": "Speed of drift (0.0 = no drift)."
            },
            {
                "name": "n_drift_centroids",
                "type": "int",
                "label": "Drifting Centroids",
                "default": 50,
                "min_value": 0,
                "step": 1,
                "help": "Number of centroids that drift."
            },
            {
                "name": "drift_width",
                "type": "int",
                "label": "Drift Width",
                "default": 400,
                "min_value": 1,
                "step": 1,
                "help": "Width of the concept drift (number of samples)."
            }
        ]

    def get_available_settings(self) -> dict[str, dict]:
        return {
            "Default": {
                "n_features": 10,
                "n_classes": 2,
                "n_centroids": 50,
                "change_speed": 0.0,
                "n_drift_centroids": 50,
                "drift_width": 400
            },
            "Fast Drift": {
                "n_features": 10,
                "n_classes": 2,
                "n_centroids": 50,
                "change_speed": 0.87,
                "n_drift_centroids": 50,
                "drift_width": 400
            }
        }

    def generate(self, n_samples_before=1000, n_samples_after=1000,
                 n_features=10, n_classes=2, n_centroids=50,
                 change_speed=0.0, n_drift_centroids=50, drift_width=400,
                 random_seed=42, **kwargs):
        """
        Generate synthetic data stream using River's RandomRBFDrift generator.
        """
        
        # Stream 1: Static (change_speed=0)
        stream_static = synth.RandomRBFDrift(
            seed_model=random_seed,
            seed_sample=random_seed,
            n_classes=n_classes,
            n_features=n_features,
            n_centroids=n_centroids,
            change_speed=0.0, # Static
            n_drift_centroids=n_drift_centroids
        )

        # Stream 2: Drifting
        # We use a different seed or same seed? 
        # If we use same seed, it starts same but drifts?
        # Hyperplane uses same seed.
        stream_drift = synth.RandomRBFDrift(
            seed_model=random_seed,
            seed_sample=random_seed,
            n_classes=n_classes,
            n_features=n_features,
            n_centroids=n_centroids,
            change_speed=change_speed,
            n_drift_centroids=n_drift_centroids
        )

        stream = synth.ConceptDriftStream(
            stream=stream_static,
            drift_stream=stream_drift,
            position=n_samples_before,
            width=drift_width,
            seed=random_seed
        )

        return generate_river_data(stream, n_samples_before, n_samples_after, n_features)
