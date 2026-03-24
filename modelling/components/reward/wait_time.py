"""Raw halting-count reward: returns negative number of stopped vehicles."""

from typing import Any
from .base import BaseReward


class WaitTimeReward(BaseReward):
    """Negative halting vehicle count across controlled lanes. Less negative = better."""

    def __init__(self, normalise: bool = True, scale: float = 1.0):
        self._normalise = normalise
        self._scale     = scale

    def compute(self, traci: Any, tls_id: str) -> float:
        """Sum halting vehicles across all controlled lanes, return negated."""
        lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))
        if not lanes:
            return 0.0

        total_halting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in lanes
        )
        if self._normalise:
            total_halting /= len(lanes)

        return -total_halting * self._scale

    def reset(self) -> None:
        pass
