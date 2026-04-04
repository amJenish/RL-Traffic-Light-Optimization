"""Composite reward: alpha * delta + (1-alpha) * pressure. Anti-oscillation by design."""

from typing import Any
from .base import BaseReward


class CompositeReward(BaseReward):
    """Blends directional delta (did queues shrink?) with absolute pressure (how bad is load?).
    Prevents the agent from gaming pure delta by oscillating phases."""

    def __init__(
        self,
        normalise: bool  = True,
        scale:     float = 1.0,
        alpha:     float = 0.65,
    ):
        self._normalise = normalise
        self._scale     = scale
        self._alpha     = alpha
        self._prev_halting: dict[str, float] = {}
        self._initialized: set[str] = set()

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        """alpha * (prev - current) + (1 - alpha) * (-current), scaled."""
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

        if tls_id not in self._initialized:
            self._initialized.add(tls_id)
            self._prev_halting[tls_id] = current_halting
            return -current_halting * (1 - self._alpha) * self._scale

        prev = self._prev_halting[tls_id]
        self._prev_halting[tls_id] = current_halting

        delta    = prev - current_halting
        pressure = -current_halting

        return (self._alpha * delta + (1 - self._alpha) * pressure) * self._scale

    def reset(self) -> None:
        self._prev_halting.clear()
        self._initialized.clear()
