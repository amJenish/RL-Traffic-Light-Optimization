"""
modeling/components/reward/base.py
------------------------------------
Abstract base class for all reward functions.

A reward function takes the current simulation state via TraCI and
returns a scalar reward signal. Swap this to change what the agent
optimises for without touching any other component.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseReward(ABC):

    @abstractmethod
    def compute(self, traci: Any, tls_id: str) -> float:
        """
        Compute and return the reward for a single traffic light
        at the current simulation step.

        Args:
            traci:   Active TraCI connection to query simulation state.
            tls_id:  Traffic light ID to compute reward for.

        Returns:
            float — scalar reward signal.
            Convention: higher is better.
            Penalise waiting → return negative values.
            Reward throughput → return positive values.
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode.
        Override if reward function maintains internal state
        (e.g. vehicle sets for throughput delta calculation).
        Default implementation does nothing.
        """
        pass