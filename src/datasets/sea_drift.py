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

    def get_available_settings(self) -> dict[str, dict]:
        return {
            "Default": {}
        }

    def generate(self, n_samples_before=1000, n_samples_after=1000, random_seed=42, **kwargs):
        """
        Generate synthetic data stream using River's SEA generator.
        Drifts from variant 0 to variant 3 at the drift point.
        The SEA generator has 3 features, but we only use the first 2.
        """
        stream_SEA = synth.ConceptDriftStream(
            stream=synth.SEA(seed=random_seed, variant=0),
            drift_stream=synth.SEA(seed=random_seed, variant=3),
            position=n_samples_before,
            width=400,  # Gradual drift
            seed=random_seed
        )
        return generate_river_data(stream_SEA, n_samples_before, n_samples_after)
