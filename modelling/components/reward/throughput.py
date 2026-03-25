"""Throughput reward: counts vehicles that cleared the intersection since the last decision."""

from typing import Any
from .base import BaseReward


class ThroughputReward(BaseReward):
    """Positive reward proportional to how many vehicles departed the network.
    For a single-intersection setup, every departed vehicle has cleared the intersection."""

    def __init__(
        self,
        normalise: bool  = True,
        scale:     float = 1.0,
        **kwargs,
    ):
        self._normalise = normalise
        self._scale     = scale
        self._prev_total: dict[str, int] = {}

    def compute(self, traci: Any, tls_id: str) -> float:
        """Delta of cumulative departed vehicles between calls, scaled."""
        current_total = traci.simulation.getArrivedNumber()

        if tls_id not in self._prev_total:
            self._prev_total[tls_id] = current_total
            return 0.0

        throughput = current_total - self._prev_total[tls_id]
        self._prev_total[tls_id] = current_total

        if self._normalise:
            lanes = list(dict.fromkeys(
                traci.trafficlight.getControlledLanes(tls_id)
            ))
            if lanes:
                throughput /= len(lanes)

        return throughput * self._scale

    def reset(self) -> None:
        self._prev_total.clear()
