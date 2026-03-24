"""Abstract base for environment wrappers. Manages the simulation lifecycle."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):

    @abstractmethod
    def start(self, route_file: str) -> None:
        """Launch a new episode with the given route file."""
        ...

    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n steps."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """True when the episode has finished."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Shut down the simulation and free resources."""
        ...

    @abstractmethod
    def get_tls_ids(self) -> list[str]:
        """All traffic light IDs in the network."""
        ...

    @abstractmethod
    def get_sim_time(self) -> float:
        """Current simulation time in seconds."""
        ...

    @property
    @abstractmethod
    def traci(self) -> Any:
        """Raw TraCI connection for observations and rewards."""
        ...
