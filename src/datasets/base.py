from abc import ABC, abstractmethod
import numpy as np

class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the dataset."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the dataset."""
        pass

    @abstractmethod
    def generate(self, **kwargs) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
        """
        Generate the dataset.

        Returns
        -------
        tuple
            (X, y, drift_point, feature_names)
        """
        pass

    def get_params(self) -> dict:
        """
        Return default parameters for the dataset.
        Can be overridden by subclasses to provide specific parameters for the UI.
        """
        return {
            "n_samples_before": 1000,
            "n_samples_after": 1000,
            "random_seed": 42
        }
