import numpy as np
import pandas as pd

from enum import Enum
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ScalingType(Enum):
    MinMax = 'min_max'
    Standard = 'standard'


class DataScaler:
    """
    Wrapper around sklearn scalers with a unified interface.

    Parameters
    ----------
    scaling_type : ScalingType
        Type of scaling to apply.

    Attributes
    ----------
    scaler : TransformerMixin
        The scaler instance.
    _is_fitted : bool
        Flag indicating if the scaler has been fitted.
    """

    scaler: TransformerMixin
    _is_fitted: bool

    def __init__(self, scaling_type: ScalingType):
        if scaling_type == ScalingType.MinMax:
            self.scaler = MinMaxScaler()
        elif scaling_type == ScalingType.Standard:
            self.scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling type")

        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the scaler on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        """
        self.scaler.fit(X)
        self._is_fitted = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Scaled data.

        Notes
        -----
            One scaler can be fitted once but used many times

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        self.fit(X)
        return self.transform(X)
