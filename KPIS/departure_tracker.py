"""Shared departure-lane counting (ThroughputReward semantics, episode-long sum)."""

from __future__ import annotations

from typing import Any

from modelling.components.reward.throughput import _departure_lanes


class DepartureThroughputTracker:
    """Per-step new vehicle IDs on departure lanes; never pops until ``reset``."""

    def __init__(self) -> None:
        self._prev_on_departure: dict[str, frozenset[str]] = {}
        self._accumulated: dict[str, int] = {}
        self._lanes_cache: dict[str, list[str]] = {}

    def reset(self) -> None:
        self._prev_on_departure.clear()
        self._accumulated.clear()
        self._lanes_cache.clear()

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
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

    def episode_departure_total(self) -> float:
        return float(sum(self._accumulated.values()))
