"""Throughput reward: vehicles that entered a departure lane from the junction this step."""

from typing import Any

from .base import BaseReward


def _departure_lanes(traci: Any, tls_id: str) -> list[str]:
    """Lanes on edges that leave the junction (same id as tlLogic, e.g. J_centre)."""
    lanes: list[str] = []
    for eid in traci.edge.getIDList():
        try:
            if traci.edge.getFromID(eid) != tls_id:
                continue
        except (traci.exceptions.TraCIException, AttributeError):
            continue
        try:
            n = traci.edge.getLaneNumber(eid)
        except traci.exceptions.TraCIException:
            continue
        for i in range(n):
            lanes.append(f"{eid}_{i}")
    return lanes


class ThroughputReward(BaseReward):
    """Reward proportional to vehicles that crossed onto outbound lanes this simulation step.

    Counts vehicles whose id was not present on any departure lane after the previous
    reward call but is present now — i.e. they entered a post-intersection lane during
    the last TraCI step. This matches “passed the intersection” rather than trip end
    (``simulation.getArrivedNumber``).
    """

    def __init__(
        self,
        normalise: bool = True,
        scale: float = 1.0,
        **kwargs,
    ):
        self._normalise = normalise
        self._scale = scale
        self._prev_on_departure: dict[str, frozenset[str]] = {}
        self._lanes_cache: dict[str, list[str]] = {}

    def compute(self, traci: Any, tls_id: str) -> float:
        if tls_id not in self._lanes_cache:
            self._lanes_cache[tls_id] = _departure_lanes(traci, tls_id)

        dep_lanes = self._lanes_cache[tls_id]
        current: set[str] = set()
        for lane in dep_lanes:
            try:
                current.update(traci.lane.getLastStepVehicleIDs(lane))
            except traci.exceptions.TraCIException:
                continue

        current_f = frozenset(current)
        if tls_id not in self._prev_on_departure:
            self._prev_on_departure[tls_id] = current_f
            return 0.0

        passed = len(current - set(self._prev_on_departure[tls_id]))
        self._prev_on_departure[tls_id] = current_f

        if self._normalise and dep_lanes:
            passed /= len(dep_lanes)

        return passed * self._scale

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._lanes_cache.clear()
