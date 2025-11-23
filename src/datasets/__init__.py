from .base import BaseDataset
from .custom_normal import CustomNormalDataset
from .custom_3d_drift import Custom3DDriftDataset
from .sea_drift import SeaDriftDataset
from .hyperplane_drift import HyperplaneDriftDataset
from .controlled_concept_drift import ControlledConceptDriftDataset
from .csv_dataset import CSVDataset

DATASETS = {
    d.name: d for d in [
        CustomNormalDataset(),
        Custom3DDriftDataset(),
        SeaDriftDataset(),
        HyperplaneDriftDataset(),
        ControlledConceptDriftDataset(),
        CSVDataset()
    ]
}

def get_dataset(name: str) -> BaseDataset:
    return DATASETS.get(name)

def get_all_datasets() -> list[BaseDataset]:
    return list(DATASETS.values())
