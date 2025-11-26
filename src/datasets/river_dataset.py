import numpy as np

from enum import Enum
from river import datasets
from .base import BaseDataset
from .utils import generate_river_data


class RiverDatasetType(Enum):
    ELECTRICITY = "Electricity"
    AIRLIENES = "Airlines"
    FOREST_COVER_TYPE = "Forest"


class RiverDataset(BaseDataset):
    def __init__(self, name: str = RiverDatasetType.ELECTRICITY.value):
        self.dataset_name = name
        if self.dataset_name == RiverDatasetType.ELECTRICITY.value:
            self.dataset = datasets.Elec2()
        elif self.dataset_name == RiverDatasetType.AIRLIENES.value:
            self.dataset = datasets.Airlines()
        elif self.dataset_name == RiverDatasetType.FOREST_COVER_TYPE.value:
            self.dataset = datasets.ForestCoverType()
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    @property
    def name(self) -> str:
        return f"{self.dataset_name.lower}_drift"

    @property
    def display_name(self) -> str:
        return f"{self.dataset_name} Drift"

    def generate(
        self,
        size_of_block=2000,
        starting_point=None,
        random_seed=42,
    ):
        # TODO: number of features
        size_of_dataset = self.dataset.n_samples
        np.random.seed(random_seed)

        if starting_point is None:
            starting_point = np.random.randint(0, size_of_dataset - size_of_block)
        else:
            starting_point = max(0, min(starting_point, size_of_dataset - size_of_block))

        stream = self.dataset.skip(starting_point)

        return generate_river_data(stream, size_of_block, n_features=len(self.dataset.features))
