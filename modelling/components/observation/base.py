"""
modeling/components/observation/base.py
-----------------------------------------
Abstract base class for all observation builders.

An observation builder takes the current simulation state via TraCI
and returns a feature vector the policy uses to make decisions.
Swap this to change what the agent sees.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseObservation(ABC):

    @abstractmethod
    def build(self, traci: Any, tls_id: str) -> np.ndarray:
        """
        Build and return the observation vector for a single traffic light.

        Args:
            traci:   Active TraCI connection to query simulation state.
            tls_id:  Traffic light ID to build observation for.

        Returns:
            np.ndarray of dtype float32. Shape must be consistent across
            all calls — the policy's input layer is sized from this.
        """
        ...

    @abstractmethod
    def size(self) -> int:
        """
        Returns the length of the observation vector.
        Called once at startup so the policy can size its input layer.
        """
        ...