"""Abstract base for learning rate schedulers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseScheduler(ABC):

    @abstractmethod
    def step(self) -> None:
        """Advance the schedule by one update step."""
        ...

    @abstractmethod
    def get_lr(self) -> float:
        """Current learning rate."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Scheduler state for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        ...
