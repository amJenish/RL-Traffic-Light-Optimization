from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from KPIS.departure_tracker import DepartureThroughputTracker
from KPIS.neg_waiting import NegLaneWaitingIntegralKpi
from KPIS.throughput import ThroughputKpi


@dataclass(frozen=True)
class KPIResult:
    label: str
    kpi_throughput_total: float
    kpi_throughput_rate: float
    kpi_neg_lane_waiting_integral: float
    phase_switch_count: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kpi_throughput_total": self.kpi_throughput_total,
            "kpi_throughput_rate": self.kpi_throughput_rate,
            "kpi_neg_lane_waiting_integral": self.kpi_neg_lane_waiting_integral,
            "phase_switch_count": self.phase_switch_count,
        }


class KPICollector:
    def __init__(self, tls_id: str) -> None:
        self._tls_id = tls_id
        self._tracker = DepartureThroughputTracker()
        self._tp = ThroughputKpi(self._tracker)
        self._neg = NegLaneWaitingIntegralKpi()

    def reset(self) -> None:
        self._tracker.reset()
        self._neg.reset()

    def on_step(self, traci: Any, *, accumulate: bool = True) -> None:
        tid = self._tls_id
        self._tracker.on_simulation_step(traci, tid, accumulate=accumulate)
        self._neg.on_simulation_step(traci, tid, accumulate=accumulate)

    def collect(
        self,
        elapsed_s: float,
        *,
        label: str,
        phase_switch_count: int,
    ) -> KPIResult:
        metrics: dict[str, Any] = {}
        self._tp.contribute_episode_metrics(metrics, elapsed_s)
        self._neg.contribute_episode_metrics(metrics, elapsed_s)
        return KPIResult(
            label=label,
            kpi_throughput_total=float(metrics.get("kpi_throughput_total", 0.0)),
            kpi_throughput_rate=float(metrics.get("kpi_throughput_rate", 0.0)),
            kpi_neg_lane_waiting_integral=float(
                metrics.get("kpi_neg_lane_waiting_integral", 0.0)
            ),
            phase_switch_count=int(phase_switch_count),
        )


def kpi_result_to_json(r: KPIResult) -> dict[str, Any]:
    return r.to_json_dict()
