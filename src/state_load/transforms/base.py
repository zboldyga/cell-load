# benchmark/data/transforms/base.py
from abc import ABC, abstractmethod
import anndata as ad
import numpy as np


class BaseTransform(ABC):
    """Base class for all data transforms."""

    @abstractmethod
    def name(self) -> str:
        """Return a string name for the transform."""
        pass

    @abstractmethod
    def fit(self, adata: ad.AnnData) -> None:
        """Fit the transform using the training data."""
        pass

    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Transform data into the target space."""
        pass

    @abstractmethod
    def decode(self, data: np.ndarray) -> np.ndarray:
        """Transform data back to gene space."""
        pass
