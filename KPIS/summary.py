"""Aggregate test-episode KPIs and write ``results.csv`` (main + grid search)."""

from __future__ import annotations

import csv
import statistics
from typing import Any


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def aggregate_test_kpis(
    test_log: list[dict[str, Any]],
    elapsed_s: float,
) -> dict[str, float]:
    """Mean/std across test episodes for crossings, throughput, neg waiting integral."""
    _ = elapsed_s
    cr = [
        float(m.get("kpi_crossings_rate", m.get("crossings_rate", 0)))
        for m in test_log
    ]
    ct = [
        float(m.get("kpi_crossings_total", m.get("crossings_total", 0)))
        for m in test_log
    ]
    tr = [float(m["kpi_throughput_rate"]) for m in test_log if "kpi_throughput_rate" in m]
    tt = [float(m["kpi_throughput_total"]) for m in test_log if "kpi_throughput_total" in m]
    nw = [
        float(m["kpi_neg_lane_waiting_integral"])
        for m in test_log
        if "kpi_neg_lane_waiting_integral" in m
    ]

    out: dict[str, float] = {}
    m, s = _mean_std(cr)
    out["test_crossings_rate_mean"], out["test_crossings_rate_std"] = m, s
    m, s = _mean_std(ct)
    out["test_crossings_total_mean"], out["test_crossings_total_std"] = m, s
    m, s = _mean_std(tr)
    out["test_throughput_rate_mean"], out["test_throughput_rate_std"] = m, s
    m, s = _mean_std(tt)
    out["test_throughput_total_mean"], out["test_throughput_total_std"] = m, s
    m, s = _mean_std(nw)
    out["test_neg_lane_waiting_integral_mean"], out["test_neg_lane_waiting_integral_std"] = m, s
    return out


def write_results_csv(path: str, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    """Write a single-row CSV with header."""
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
