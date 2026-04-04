"""Build ``schedule.json`` from test phase-sequence exports + day CSV coverage."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.BuildRoute import time_to_seconds


def load_coverage_intervals(day_id: int, days_dir: str) -> list[tuple[int, int]]:
    """Return [begin_s, end_s) intervals from processed day CSV (demand coverage)."""
    if not days_dir:
        return []
    path = Path(days_dir) / f"day_{day_id:02d}.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    intervals: list[tuple[int, int]] = []
    for _, row in df.iterrows():
        begin = time_to_seconds(row["start_time"])
        if "end_time" in df.columns and pd.notna(row.get("end_time")):
            end = time_to_seconds(row["end_time"])
        else:
            end = begin + 15 * 60
        intervals.append((begin, end))
    return intervals


def aggregate_schedule(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by (tls_id, phase, 15-min bucket); median/std/n on duration_s."""
    groups: dict[tuple[str, int, int], list[float]] = {}
    for e in entries:
        key = (e["tls_id"], int(e["phase"]), int(e["bucket_start_s"]))
        groups.setdefault(key, []).append(float(e["duration_s"]))

    rows: list[dict[str, Any]] = []
    for (tls_id, phase, bucket_start_s), durations in groups.items():
        n = len(durations)
        med = float(statistics.median(durations))
        std = float(statistics.pstdev(durations)) if n > 1 else 0.0
        rows.append(
            {
                "tls_id": tls_id,
                "phase": phase,
                "bucket_start_s": bucket_start_s,
                "median_s": med,
                "std_s": std,
                "n": n,
            }
        )
    rows.sort(key=lambda r: (r["bucket_start_s"], r["tls_id"], r["phase"]))
    return rows


def write_schedule_json(path: str, buckets: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"buckets": buckets}, f, indent=2)


def collect_schedule_entries_from_run(
    output_dir: str,
    test_log: list[dict[str, Any]],
    days_dir: str,
    warn: Any | None = None,
) -> list[dict[str, Any]]:
    """Load ``test_sequences/episode_*.json`` + coverage for each test episode (like ``Trainer``)."""
    all_entries: list[dict[str, Any]] = []
    for ep_idx, m in enumerate(test_log, start=1):
        day_id = int(m["day_id"])
        seq_path = Path(output_dir) / "test_sequences" / f"episode_{ep_idx:03d}.json"
        if seq_path.is_file():
            with open(seq_path, encoding="utf-8") as sf:
                seq_data = json.load(sf)
            seq = seq_data.get("tls_sequences") or {}
        else:
            seq = {}
        coverage = load_coverage_intervals(day_id, days_dir)
        if not coverage:
            if warn:
                warn(f"  Warning: no coverage intervals for day {day_id} (days_dir={days_dir!r})")
            continue
        for tls_id, events in seq.items():
            for ev in events:
                st = float(ev["sim_time"])
                if not any(a <= st < b for a, b in coverage):
                    continue
                all_entries.append(
                    {
                        "tls_id": tls_id,
                        "day_id": day_id,
                        "phase": int(ev["phase"]),
                        "duration_s": float(ev["duration_s"]),
                        "sim_time": st,
                        "bucket_start_s": 900 * (int(st) // 900),
                    }
                )
    return all_entries


def write_test_sequence_episode_json(
    output_dir: str,
    episode_idx: int,
    tls_sequences: dict[str, list[dict[str, Any]]],
) -> None:
    seq_dir = Path(output_dir) / "test_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    path = seq_dir / f"episode_{episode_idx:03d}.json"
    payload = {"episode": episode_idx, "tls_sequences": tls_sequences}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_schedule_for_run(
    output_dir: str,
    test_log: list[dict[str, Any]],
    days_dir: str,
    warn: Any | None = None,
) -> str | None:
    """Aggregate and write ``schedule.json`` under ``output_dir``; return path or None."""
    entries = collect_schedule_entries_from_run(output_dir, test_log, days_dir, warn=warn)
    if not entries:
        return None
    schedule = aggregate_schedule(entries)
    path = str(Path(output_dir) / "schedule.json")
    write_schedule_json(path, schedule)
    return path
