"""Composite throughput reward: reward exits, penalize queue pressure."""

from typing import Any

from .base import BaseReward
from .throughput import _departure_lanes


class ThroughputQueueReward(BaseReward):
    """Reward total exits over interval minus current queue pressure.

    Designed for SMDP-style decision intervals:
    - :meth:`on_simulation_step` accumulates throughput after every simulation step.
    - :meth:`compute` is called at decision epochs to finalize interval reward.
    """

    def __init__(
        self,
        normalise: bool = True,
        scale: float = 1.0,
        throughput_weight: float = 1.0,
        queue_weight: float = 1.0,
        switch_weight: float = 0.5,
        **kwargs: Any,
    ):
        self._normalise = normalise
        self._scale = scale
        self._throughput_weight = throughput_weight
        self._queue_weight = queue_weight
        self._switch_weight = switch_weight

        self._prev_on_departure: dict[str, frozenset[str]] = {}
        self._accumulated_throughput: dict[str, int] = {}
        self._departure_lanes_cache: dict[str, list[str]] = {}

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        """Track newly observed vehicles on departure lanes at each SUMO step."""
        if tls_id not in self._departure_lanes_cache:
            self._departure_lanes_cache[tls_id] = _departure_lanes(traci, tls_id)

        dep_lanes = self._departure_lanes_cache[tls_id]
        current: set[str] = set()
        for lane in dep_lanes:
            try:
                current.update(traci.lane.getLastStepVehicleIDs(lane))
            except traci.exceptions.TraCIException:
                continue

        current_f = frozenset(current)
        if tls_id not in self._prev_on_departure:
            self._prev_on_departure[tls_id] = current_f
            return

        prev = self._prev_on_departure[tls_id]
        passed = len(current - set(prev))
        self._prev_on_departure[tls_id] = current_f

        if accumulate:
            self._accumulated_throughput[tls_id] = (
                self._accumulated_throughput.get(tls_id, 0) + passed
            )

    def compute(
        self, traci: Any, tls_id: str, *, switched: bool = False
    ) -> float:
        """Finalize interval reward: weighted throughput minus weighted queue."""
        throughput = float(self._accumulated_throughput.pop(tls_id, 0))

        controlled_lanes = list(dict.fromkeys(
            traci.trafficlight.getControlledLanes(tls_id)
        ))
        queue = 0.0
        if controlled_lanes:
            queue = float(sum(
                traci.lane.getLastStepHaltingNumber(lane)
                for lane in controlled_lanes
            ))

        if self._normalise:
            dep_lanes = self._departure_lanes_cache.get(tls_id, [])
            if dep_lanes:
                throughput /= len(dep_lanes)
            if controlled_lanes:
                queue /= len(controlled_lanes)

        reward = self._throughput_weight * throughput - self._queue_weight * queue
        reward -= self._switch_weight * float(switched)
        return reward * self._scale

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._accumulated_throughput.clear()
        self._departure_lanes_cache.clear()
