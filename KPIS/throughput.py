"""Throughput KPI: same departure count as ThroughputReward (shared tracker)."""

from __future__ import annotations

from typing import Any

from .base import EpisodeKpi
from .departure_tracker import DepartureThroughputTracker


class ThroughputKpi(EpisodeKpi):
    """Reads the shared :class:`DepartureThroughputTracker`; no duplicate TraCI work."""

    def __init__(self, tracker: DepartureThroughputTracker) -> None:
        self._tracker = tracker

    def reset(self) -> None:
        pass

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        pass

    def contribute_episode_metrics(
        self, metrics: dict[str, Any], elapsed_s: float
    ) -> None:
        total = self._tracker.episode_departure_total()
        rate = float(total) / max(1e-9, elapsed_s)
        metrics["kpi_throughput_total"] = total
        metrics["kpi_throughput_rate"] = rate
