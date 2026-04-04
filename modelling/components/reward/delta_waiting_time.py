"""Differential reward: previous waiting-time metric minus current (per decision)."""

from typing import Any

from .base import BaseReward


class DeltaWaitingTimeReward(BaseReward):
    """Change in per-step lane waiting time between consecutive decisions.

    Uses SUMO ``lane.getWaitingTime`` summed over controlled lanes at each decision
    (same idea as :class:`DeltaVehicleCountReward`, but for waiting time).
    """

    def __init__(self, normalise: bool = True, scale: float = 1.0):
        self._normalise = normalise
        self._scale = scale
        self._prev_wait: dict[str, float] = {}

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        if not lanes:
            return 0.0

        current = 0.0
        for lane in lanes:
            try:
                current += float(traci.lane.getWaitingTime(lane))
            except traci.exceptions.TraCIException:
                continue
        if self._normalise:
            current /= len(lanes)

        prev = self._prev_wait.get(tls_id, current)
        self._prev_wait[tls_id] = current

        return (prev - current) * self._scale

    def reset(self) -> None:
        self._prev_wait.clear()
