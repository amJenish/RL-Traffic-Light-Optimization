"""Negative cumulative lane waiting time (larger is better)."""

from __future__ import annotations

from typing import Any

from .base import EpisodeKpi


class NegLaneWaitingIntegralKpi(EpisodeKpi):
    """Sums SUMO ``lane.getWaitingTime`` over controlled lanes each step; reports negative sum."""

    def __init__(self) -> None:
        self._sum: float = 0.0

    def reset(self) -> None:
        self._sum = 0.0

    def on_simulation_step(
        self, traci: Any, tls_id: str, *, accumulate: bool = True
    ) -> None:
        if not accumulate:
            return
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
        step_sum = 0.0
        for lane in lanes:
            try:
                step_sum += float(traci.lane.getWaitingTime(lane))
            except traci.exceptions.TraCIException:
                continue
        self._sum += step_sum

    def contribute_episode_metrics(
        self, metrics: dict[str, Any], elapsed_s: float
    ) -> None:
        _ = elapsed_s
        metrics["kpi_neg_lane_waiting_integral"] = float(-self._sum)
