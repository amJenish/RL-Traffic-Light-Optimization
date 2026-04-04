"""Negative cumulative waiting time across controlled lanes per decision interval."""

from typing import Any

from .base import BaseReward


class WaitingTimeReward(BaseReward):
    """Reward based on negative cumulative waiting time across controlled lanes.

    Accumulates per-lane waiting time at every simulation step, then returns
    the negative normalised sum at each decision epoch.

    Args:
        normalise: If True, divide accumulated waiting time by number of lanes.
        scale: Scalar multiplier applied to the final reward.
        switch_weight: Penalty subtracted when the agent switches phase.
    """

    def __init__(
        self,
        normalise: bool = True,
        scale: float = 1.0,
        switch_weight: float = 0.5,
        **kwargs: Any,
    ):
        self._normalise = normalise
        self._scale = scale
        self._switch_weight = switch_weight
        self._accumulated_waiting: dict[str, float] = {}

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        step_sum = 0.0
        for lane in lanes:
            try:
                step_sum += float(traci.lane.getWaitingTime(lane))
            except traci.exceptions.TraCIException:
                continue
        if accumulate:
            self._accumulated_waiting[tls_id] = (
                self._accumulated_waiting.get(tls_id, 0.0) + step_sum
            )

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        total = float(self._accumulated_waiting.pop(tls_id, 0.0))
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        if self._normalise and lanes:
            total /= len(lanes)
        reward = -total
        reward -= self._switch_weight * float(switched)
        return float(reward * self._scale)

    def reset(self) -> None:
        self._accumulated_waiting.clear()
