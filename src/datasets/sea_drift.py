from river.datasets import synth
from .base import BaseDataset
from .utils import generate_river_data


class SeaDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "sea_drift"

    @property
    def display_name(self) -> str:
        return "SEA Drift"

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

    def get_available_settings(self) -> dict[str, dict]:
        return {
            "Default": {}
        }

    def generate(self, n_samples_before=1000, n_samples_after=1000, random_seed=42, **kwargs):
        """
        Generate synthetic data stream using River's SEA generator.
        Drifts from variant 0 to variant 3 at the drift point.
        The SEA generator has 3 features, and we use all of them.
        """
        stream_SEA = synth.ConceptDriftStream(
            stream=synth.SEA(seed=random_seed, variant=0),
            drift_stream=synth.SEA(seed=random_seed, variant=3),
            position=n_samples_before,
            width=400,  # Gradual drift
            seed=random_seed
        )
        return generate_river_data(stream_SEA, n_samples_before + n_samples_after, n_features=3)
