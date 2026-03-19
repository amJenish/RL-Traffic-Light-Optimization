"""
modeling/components/environment/base.py
----------------------------------------
Abstract base class for all environment implementations.

An environment wraps the traffic simulation and exposes a standard
step/reset interface that the Agent uses. Swap this to change the
underlying simulator (e.g. SUMO, mock, replay).
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):

    @abstractmethod
    def start(self, route_file: str) -> None:
        """
        Start or restart the simulation for a new episode.

        Args:
            route_file: Path to the SUMO .rou.xml flow file for this episode.
        """
        ...

    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        """
        Advance the simulation by n_steps seconds.

        Args:
            n_steps: Number of simulation seconds to advance.
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """
        Returns True when the episode is complete
        (no more vehicles expected in simulation).
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """
        Cleanly shut down the simulation and release all resources.
        Must be called at the end of every episode.
        """
        ...

    @abstractmethod
    def get_tls_ids(self) -> list[str]:
        """
        Returns the list of all traffic light IDs in the simulation.
        Used by the Agent to know which junctions to control.
        """
        ...

    @abstractmethod
    def get_sim_time(self) -> float:
        """
        Returns current simulation time in seconds.
        """
        ...

    @property
    @abstractmethod
    def traci(self) -> Any:
        """
        Exposes the raw TraCI connection so observation builders
        and reward functions can query simulation state directly.
        """
        ...