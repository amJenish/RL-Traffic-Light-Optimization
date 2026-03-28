"""Throughput reward: vehicles that enter departure lanes (post-intersection) per SUMO step."""

from typing import Any

from .base import BaseReward


def _departure_lanes_from_edges(traci: Any, tls_id: str) -> list[str]:
    """Lanes on edges whose start node is the TLS junction (e.g. J_centre)."""
    tls = str(tls_id)
    lanes: list[str] = []
    for eid in traci.edge.getIDList():
        try:
            if str(traci.edge.getFromID(eid)) != tls:
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


def _departure_lanes_from_links(traci: Any, tls_id: str) -> list[str]:
    """Fallback: outgoing road lanes from TLS controlled links (skip internal ':' lanes)."""
    lanes: list[str] = []
    try:
        groups = traci.trafficlight.getControlledLinks(tls_id)
    except traci.exceptions.TraCIException:
        return lanes
    for group in groups:
        for tup in group:
            if len(tup) >= 2:
                lane = tup[1]
                if lane and not str(lane).startswith(":"):
                    lanes.append(str(lane))
    return list(dict.fromkeys(lanes))


def _departure_lanes(traci: Any, tls_id: str) -> list[str]:
    primary = _departure_lanes_from_edges(traci, tls_id)
    if primary:
        return primary
    return _departure_lanes_from_links(traci, tls_id)


class ThroughputReward(BaseReward):
    """Counts total vehicle exits over the interval between decision epochs.

    The agent calls :meth:`on_simulation_step` after **every** ``simulationStep``.
    :meth:`compute` is called only at decision epochs and returns the total number of newly
    seen vehicles on departure lanes since the previous :meth:`compute` (optionally normalized
    per departure lane), times ``scale``.
    """

    def __init__(
        self,
        normalise: bool = False,
        scale: float = 1.0,
        **kwargs: Any,
    ):
        self._normalise = normalise
        self._scale = scale
        self._prev_on_departure: dict[str, frozenset[str]] = {}
        self._accumulated: dict[str, int] = {}
        self._lanes_cache: dict[str, list[str]] = {}

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        """Update crossing counts after one SUMO step. Use accumulate=False during warmup."""
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
            return

        prev = self._prev_on_departure[tls_id]
        passed = len(current - set(prev))
        self._prev_on_departure[tls_id] = current_f

        if accumulate:
            self._accumulated[tls_id] = self._accumulated.get(tls_id, 0) + passed

    def compute(self, traci: Any, tls_id: str) -> float:
        total = float(self._accumulated.pop(tls_id, 0))
        dep_lanes = self._lanes_cache.get(tls_id, [])
        if self._normalise and dep_lanes:
            total /= len(dep_lanes)
        return total * self._scale

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._accumulated.clear()
        self._lanes_cache.clear()
