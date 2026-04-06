from __future__ import annotations

from typing import Any


class ScheduleController:
    def __init__(
        self,
        schedule: dict[str, Any],
        tls_id: str,
        min_green_s: float,
        yellow_duration_s: float,
        step_length: float,
        bucket_width_s: int = 900,
    ) -> None:
        self._tls_id = tls_id
        self._min_green_s = float(min_green_s)
        self._yellow_duration_s = float(yellow_duration_s)
        self._step_length = float(step_length)
        self._bw = int(bucket_width_s)
        self._by_phase: dict[int, list[tuple[float, float]]] = {}
        for b in schedule.get("buckets") or []:
            if b.get("tls_id") != tls_id:
                continue
            ph = int(b["phase"])
            bs = float(b["bucket_start_s"])
            med = float(b["median_s"])
            self._by_phase.setdefault(ph, []).append((bs, med))
        for ph in self._by_phase:
            self._by_phase[ph].sort(key=lambda x: x[0])

    def get_action(
        self,
        current_green_phase: int,
        sim_time: float,
        time_in_phase: float,
    ) -> int:
        ph = int(current_green_phase)
        bucket = (int(sim_time) // self._bw) * float(self._bw)
        entries = self._by_phase.get(ph)
        if not entries:
            return 0
        exact = [e for e in entries if abs(e[0] - bucket) < 1e-3]
        if exact:
            target_s = exact[0][1]
        else:
            best = min(entries, key=lambda e: abs(e[0] - bucket))
            target_s = best[1]
        return 1 if time_in_phase >= target_s else 0
