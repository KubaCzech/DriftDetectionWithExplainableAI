from .base import BaseDataset
from .sea_drift import SeaDriftDataset
from .hyperplane_drift import HyperplaneDriftDataset
from .controlled_concept_drift import ControlledConceptDriftDataset
from .csv_dataset import CSVDataset
from .random_rbf_drift import RandomRBFDriftDataset
from .sdbm_rbf_drift import SDBMRBFDriftDataset

from .river_dataset import RiverDataset, RiverDatasetType

DATASETS = {
    d.name: d for d in [
        SeaDriftDataset(),
        HyperplaneDriftDataset(),
        ControlledConceptDriftDataset(),
        RandomRBFDriftDataset(),
        SDBMRBFDriftDataset(),
        CSVDataset(),
        RiverDataset(RiverDatasetType.ELECTRICITY.value),
        # RiverDataset(RiverDatasetType.AIRLIENES.value),
        # RiverDataset(RiverDatasetType.FOREST_COVER_TYPE.value)
    ]
}


def get_dataset(name: str) -> BaseDataset:
    return DATASETS.get(name)


def get_all_datasets() -> list[BaseDataset]:
    return list(DATASETS.values())
