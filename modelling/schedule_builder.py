"""Aggregate green-phase duration records into schedule.json buckets."""

from __future__ import annotations

import json
import statistics
import warnings
from typing import Any


class ScheduleBuilder:
    def __init__(self, bucket_width_s: int = 900) -> None:
        self._w = int(bucket_width_s)

    def build(
        self,
        records: list[dict[str, Any]],
        tls_id: str,
        *,
        min_green_s: float = 0.0,
    ) -> dict[str, Any]:
        """
        Group records by (tls_id, phase, bucket_start_s) where
        bucket_start = floor(end_s / width) * width.
        """
        w = self._w
        groups: dict[tuple[str, int, float], list[float]] = {}
        for r in records:
            if r.get("tls_id") != tls_id:
                continue
            phase = int(r["phase"])
            end_s = float(r["end_s"])
            bucket_start = (int(end_s) // w) * float(w)
            dur = float(r["duration_s"])
            key = (tls_id, phase, bucket_start)
            groups.setdefault(key, []).append(dur)

        rows: list[dict[str, Any]] = []
        for (tid, phase, bucket_start_s), durs in groups.items():
            n = len(durs)
            med = float(statistics.median(durs))
            std = float(statistics.pstdev(durs)) if n > 1 else 0.0
            if min_green_s > 0.0 and med < min_green_s:
                warnings.warn(
                    f"ScheduleBuilder: median_s {med:.3f} < min_green_s {min_green_s} "
                    f"(phase={phase}, bucket={bucket_start_s}); clamping.",
                    stacklevel=1,
                )
                med = float(min_green_s)
            rows.append(
                {
                    "tls_id": tid,
                    "phase": phase,
                    "bucket_start_s": bucket_start_s,
                    "median_s": med,
                    "std_s": std,
                    "n": n,
                }
            )
        rows.sort(key=lambda r: (r["phase"], r["bucket_start_s"], r["tls_id"]))
        return {"buckets": rows}

    def write(self, schedule: dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schedule, f, indent=2)
