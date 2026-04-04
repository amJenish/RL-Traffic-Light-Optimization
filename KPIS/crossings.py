"""Crossings / departure KPI (leaderboard-compatible naming)."""

from __future__ import annotations

from typing import Any

from .base import EpisodeKpi
from .departure_tracker import DepartureThroughputTracker


class CrossingsKpi(EpisodeKpi):
    """Owns the shared departure tracker: reset + per-step updates."""

    def __init__(self, tracker: DepartureThroughputTracker) -> None:
        self._tracker = tracker

    def reset(self) -> None:
        self._tracker.reset()

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        self._tracker.on_simulation_step(traci, tls_id, accumulate=accumulate)

    def contribute_episode_metrics(
        self, metrics: dict[str, Any], elapsed_s: float
    ) -> None:
        total = self._tracker.episode_departure_total()
        rate = float(total) / max(1e-9, elapsed_s)
        metrics["kpi_crossings_total"] = total
        metrics["kpi_crossings_rate"] = rate
        metrics["crossings_total"] = total
        metrics["crossings_rate"] = rate
