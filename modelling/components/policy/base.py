"""Abstract base for all policies. Maps observations to discrete actions."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BasePolicy(ABC):

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Pick an action given the current observation. 0 = keep, 1 = switch."""
        ...

    @abstractmethod
    def update(self, replay_buffer: Any) -> float | None:
        """One learning step from the replay buffer. Returns loss or None."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist weights and training state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore weights and training state from disk."""
        ...

    def set_eval_mode(self) -> None:
        """Disable exploration."""
        pass

    def set_train_mode(self) -> None:
        """Re-enable exploration."""
        pass

    @property
    def optimizer(self) -> Any:
        """Expose the optimizer so a scheduler can attach to it."""
        return getattr(self, "_optimiser", None)
