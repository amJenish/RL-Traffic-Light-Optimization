"""Webster (1958) cycle length and green splits from test-day count CSVs → schedule buckets."""

from __future__ import annotations

import json
import statistics
import warnings
from collections import defaultdict
from typing import Any

import pandas as pd

from preprocessing.BuildNetwork import build_phases
from preprocessing.BuildRoute import time_to_seconds

SAT_VPH_PER_LANE = 1800.0
STARTUP_LOST_S = 1.0


class WebsterScheduleBuilder:
    def __init__(
        self,
        intersection_cfg: dict[str, Any],
        bucket_width_s: int = 900,
    ) -> None:
        self._ic = intersection_cfg
        self._bucket_w = int(bucket_width_s)

    def _green_phases(self) -> list[dict[str, Any]]:
        ic = self._ic
        active = list(ic["active_approaches"])
        mode = str(ic.get("phases", "2"))
        amber_s = int(float(ic.get("amber_s", 3)))
        min_green_s = int(float(ic.get("min_green_s", 15)))
        cycle_s = int(float(ic.get("cycle_s", 120)))
        approaches = ic.get("approaches") or {}
        lt = str(ic.get("left_turn_mode", "permissive"))
        prot = ic.get("protected_approaches")
        if not isinstance(prot, list):
            prot = None
        return build_phases(
            active,
            mode,
            amber_s,
            min_green_s,
            cycle_s,
            approaches=approaches,
            left_turn_mode=lt,
            protected_approaches=prot,
        )

    @staticmethod
    def _lane_count(approaches: dict, d: str, m: str) -> int:
        app = approaches.get(d) or {}
        lanes = app.get("lanes") or {}
        return max(0, int(lanes.get(m, 0) or 0))

    def _compute_slot_greens(
        self,
        row: pd.Series,
        phases: list[dict[str, Any]],
        slot_minutes: int,
        approaches: dict,
        amber_s: float,
        min_green_s: float,
        max_green_s: float,
    ) -> list[float]:
        """Webster greens g_i (seconds) for one demand row."""
        vph_scale = 60.0 / float(slot_minutes)
        n_phases = len(phases)
        L = float(n_phases) * (float(amber_s) + STARTUP_LOST_S)
        y_vals: list[float] = []
        for ph in phases:
            groups = ph.get("green_groups") or []
            best = 0.0
            for d, m in groups:
                q = float(row.get(f"{d}_{m}", 0) or 0) * vph_scale
                n_l = self._lane_count(approaches, d, m)
                s = SAT_VPH_PER_LANE * max(n_l, 1)
                ratio = q / s if s > 0 else 0.0
                best = max(best, ratio)
            y_vals.append(best)
        y_sum = sum(y_vals)
        if y_sum < 1e-6:
            return [float(min_green_s)] * n_phases
        if y_sum >= 1.0:
            warnings.warn(
                "WebsterScheduleBuilder: Y >= 1.0 (oversaturated); using max_green_s "
                "for all phases for this slot.",
                stacklevel=2,
            )
            return [float(max_green_s)] * n_phases

        c_low = float(n_phases) * float(min_green_s) + L
        c_high = float(n_phases) * float(max_green_s) + L
        c_opt = (1.5 * L + 5.0) / max(1e-9, 1.0 - y_sum)
        c = max(c_low, min(c_opt, c_high))
        g_eff = c - L
        greens = [g_eff * (yi / y_sum) for yi in y_vals]
        out = [
            max(float(min_green_s), min(float(max_green_s), float(g)))
            for g in greens
        ]
        return out

    def build_from_day_csvs(
        self,
        day_csv_paths: list[str],
        tls_id: str,
        *,
        slot_minutes: int = 15,
    ) -> dict[str, Any]:
        ic = self._ic
        approaches = ic.get("approaches") or {}
        min_green_s = float(ic.get("min_green_s", 15))
        max_green_s = float(ic.get("max_green_s", 90))
        amber_s = float(ic.get("amber_s", 3))
        cycle_s = float(ic.get("cycle_s", 120))
        phases = self._green_phases()
        n_phases = len(phases)
        bw = self._bucket_w

        samples: dict[tuple[int, float], list[float]] = defaultdict(list)
        for path in day_csv_paths:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                start_t = row["start_time"]
                slot_start_s = float(time_to_seconds(start_t))
                bucket_start = (int(slot_start_s) // bw) * float(bw)
                greens = self._compute_slot_greens(
                    row,
                    phases,
                    slot_minutes,
                    approaches,
                    amber_s,
                    min_green_s,
                    max_green_s,
                )
                for phase_idx, g in enumerate(greens):
                    samples[(phase_idx, bucket_start)].append(float(g))

        rows: list[dict[str, Any]] = []
        for (phase, bucket_start_s), vals in samples.items():
            n = len(vals)
            med = float(statistics.median(vals))
            std = float(statistics.pstdev(vals)) if n > 1 else 0.0
            med = max(min_green_s, min(max_green_s, med))
            rows.append(
                {
                    "tls_id": tls_id,
                    "phase": int(phase),
                    "bucket_start_s": bucket_start_s,
                    "median_s": med,
                    "std_s": std,
                    "n": n,
                }
            )

        rows.sort(key=lambda r: (r["phase"], r["bucket_start_s"], r["tls_id"]))

        by_bucket: dict[float, list[float]] = defaultdict(list)
        for r in rows:
            by_bucket[r["bucket_start_s"]].append(r["median_s"])
        for bstart, meds in by_bucket.items():
            if sum(meds) > cycle_s + 0.01:
                warnings.warn(
                    f"WebsterScheduleBuilder: sum of phase medians {sum(meds):.2f}s "
                    f"> cycle_s {cycle_s}s at bucket_start_s={bstart}.",
                    stacklevel=1,
                )

        return {"buckets": rows}

    def write(self, schedule: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2)
