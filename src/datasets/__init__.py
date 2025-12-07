from .base import BaseDataset
from .sea_drift import SeaDriftDataset
from .hyperplane_drift import HyperplaneDriftDataset
from .controlled_concept_drift import ControlledConceptDriftDataset
from .csv_dataset import CSVDataset
from .random_rbf_drift import RandomRBFDriftDataset
from .sdbm_rbf_drift import SDBMRBFDriftDataset

from .river_dataset import RiverDataset, RiverDatasetType
from .dataset_registry import DatasetRegistry
from .imported_dataset import ImportedCSVDataset

def load_datasets():
    base_datasets = [
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
    
    datasets_dict = {d.name: d for d in base_datasets}
    
    # Load imported datasets
    registry = DatasetRegistry()
    for name, info in registry.list_datasets().items():
        datasets_dict[name] = ImportedCSVDataset(name, info, registry)
        
    return datasets_dict

DATASETS = load_datasets()

def reload_datasets():
    new_datasets = load_datasets()
    DATASETS.clear()
    DATASETS.update(new_datasets)


def get_dataset(name: str) -> BaseDataset:
    return DATASETS.get(name)


def get_all_datasets() -> list[BaseDataset]:
    return list(DATASETS.values())
