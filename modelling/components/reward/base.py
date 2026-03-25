"""Abstract base for all reward functions. Higher is better."""

from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):

    @abstractmethod
    def compute(self, traci: Any, tls_id: str) -> float:
        """Compute scalar reward for one traffic light at the current step."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if stateful."""
        pass

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        """Optional hook after each TraCI simulationStep (e.g. throughput)."""
        pass
