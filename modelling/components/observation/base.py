"""Abstract base for observation builders."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseObservation(ABC):

    @abstractmethod
    def build(self, traci: Any, tls_id: str) -> np.ndarray:
        """Build a fixed-size observation vector for one traffic light."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Length of the observation vector."""
        ...
