"""Episode KPI trackers (crossings, throughput, waiting) shared by main and grid search."""

from __future__ import annotations

from typing import Sequence

from .base import EpisodeKpi
from .crossings import CrossingsKpi
from .departure_tracker import DepartureThroughputTracker
from .neg_waiting import NegLaneWaitingIntegralKpi
from .throughput import ThroughputKpi
from .summary import aggregate_test_kpis, write_results_csv


def default_episode_kpis() -> Sequence[EpisodeKpi]:
    """Crossings + throughput share one departure tracker; neg-waiting is independent."""
    tracker = DepartureThroughputTracker()
    return (
        CrossingsKpi(tracker),
        ThroughputKpi(tracker),
        NegLaneWaitingIntegralKpi(),
    )


__all__ = [
    "CrossingsKpi",
    "DepartureThroughputTracker",
    "EpisodeKpi",
    "NegLaneWaitingIntegralKpi",
    "ThroughputKpi",
    "aggregate_test_kpis",
    "default_episode_kpis",
    "write_results_csv",
]
