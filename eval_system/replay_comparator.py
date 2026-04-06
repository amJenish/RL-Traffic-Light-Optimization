from __future__ import annotations

import csv
import os
import statistics
from typing import Any

from eval_system.kpi_collector import KPIResult


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


class ReplayComparator:
    def compare(
        self,
        dqn_results: list[KPIResult],
        webster_results: list[KPIResult],
        out_dir: str,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        metrics_spec: list[tuple[str, str, str]] = [
            ("kpi_throughput_total", "Throughput total (veh)", "float"),
            ("kpi_throughput_rate", "Throughput rate (veh/s)", "float"),
            (
                "kpi_neg_lane_waiting_integral",
                "Neg lane wait integral",
                "float",
            ),
            ("phase_switch_count", "Phase switches", "info"),
        ]

        rows_out: list[dict[str, Any]] = []
        print()
        print("=" * 64)
        print(f"{'Metric':<34} {'DQN':>12} {'Webster':>12} {'Delta':>10}")
        print("-" * 64)

        dqn_wins = 0
        comparable = 0

        for attr, title, kind in metrics_spec:
            dv = [float(getattr(r, attr)) for r in dqn_results]
            wv = [float(getattr(r, attr)) for r in webster_results]
            dm, ds = _mean_std(dv)
            wm, ws = _mean_std(wv)
            delta = dm - wm
            if kind == "info":
                winner = "info"
                win_str = "—"
            else:
                comparable += 1
                if delta > 1e-9:
                    winner = "DQN"
                    dqn_wins += 1
                elif delta < -1e-9:
                    winner = "Webster"
                else:
                    winner = "tie"
                win_str = winner

            def fmt(m: float, s: float) -> str:
                if kind == "float" and abs(m) >= 1000:
                    return f"{m:,.0f}+/-{s:,.0f}"
                return f"{m:.3f}+/-{s:.3f}"

            print(
                f"{title:<34} {fmt(dm, ds):>12} {fmt(wm, ws):>12} "
                f"{delta:>+10.3f}"
            )
            rows_out.append(
                {
                    "metric": title,
                    "dqn_mean": dm,
                    "dqn_std": ds,
                    "webster_mean": wm,
                    "webster_std": ws,
                    "delta_mean": delta,
                    "winner": win_str,
                }
            )

        print("=" * 64)
        print(
            f"{'Winner (most metrics)':<34} {'DQN':>12} "
            f"{'':>12} ({dqn_wins} / {comparable})"
        )
        print()

        path = os.path.join(out_dir, "comparison.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "metric",
                    "dqn_mean",
                    "dqn_std",
                    "webster_mean",
                    "webster_std",
                    "delta_mean",
                    "winner",
                ],
            )
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
