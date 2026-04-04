"""Differential reward: previous_halting - current_halting. Positive = queues shrank."""

from typing import Any
from .base import BaseReward


class DeltaVehicleCountReward(BaseReward):
    """Change in halting count between consecutive decisions.
    Isolates the agent's action effect from background traffic volume."""

    def __init__(self, normalise: bool = True, scale: float = 1.0):
        self._normalise = normalise
        self._scale     = scale
        self._prev_halting: dict[str, float] = {}

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        """Return (previous halting) - (current halting), scaled."""
        lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))
        if not lanes:
            return 0.0

        current_halting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in lanes
        )
        if self._normalise:
            current_halting /= len(lanes)

        prev = self._prev_halting.get(tls_id, current_halting)
        self._prev_halting[tls_id] = current_halting

        return (prev - current_halting) * self._scale

    def reset(self) -> None:
        self._prev_halting.clear()
