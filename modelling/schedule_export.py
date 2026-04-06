"""Build ``schedule.json`` from test phase-sequence exports + day CSV coverage."""

from __future__ import annotations

import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.BuildRoute import time_to_seconds


def infer_tll_path_from_net(net_file: str | None) -> str | None:
    """``Foo.net.xml`` -> ``Foo.tll.xml`` in the same directory, if the file exists."""
    if not net_file:
        return None
    p = Path(net_file)
    if not p.name.endswith(".net.xml"):
        return None
    cand = p.parent / (p.name[: -len(".net.xml")] + ".tll.xml")
    return str(cand) if cand.is_file() else None


def _link_signal_summary(state: str) -> str:
    """Short decoding of SUMO phase ``state`` (one char per controlled link)."""
    if not state:
        return "No state string recorded."
    n_g = sum(1 for c in state if c == "G")
    n_gm = sum(1 for c in state if c == "g")
    n_y = sum(1 for c in state if c == "y")
    n_r = sum(1 for c in state if c == "r")
    n_o = sum(1 for c in state if c not in "Ggyr")
    parts = []
    if n_g:
        parts.append(f"{n_g} link(s) major green (G)")
    if n_gm:
        parts.append(f"{n_gm} link(s) minor green (g)")
    if n_y:
        parts.append(f"{n_y} yellow (y)")
    if n_r:
        parts.append(f"{n_r} red (r)")
    if n_o:
        parts.append(f"{n_o} other / off ({n_o} chars)")
    return "Signal links in SUMO order: " + ", ".join(parts) + "."


def _human_phase_meaning(name: str, state: str) -> str:
    """Readable description from tlLogic ``name`` (preferred) plus state summary."""
    raw = (name or "").strip()
    if not raw:
        return _link_signal_summary(state)

    nl = raw.lower()
    if "yellow_clearance" in nl:
        return (
            "TraCI change interval between green program phases; length follows "
            "yellow_duration_s in config.json. "
            + _link_signal_summary(state)
        )
    if "amber" in nl:
        return (
            "Yellow / clearance between greens: vehicles facing yellow should stop; "
            "typically short. "
            + _link_signal_summary(state)
        )
    if "service" in nl:
        if "ns" in nl:
            return (
                "North-South: through, right, and permissive left (as wired); "
                "East-West held red. "
                + _link_signal_summary(state)
            )
        if "ew" in nl:
            return (
                "East-West: through, right, and permissive left (as wired); "
                "North-South held red. "
                + _link_signal_summary(state)
            )

    approach = ""
    if nl.startswith("ns") or nl.startswith("sn"):
        approach = "North-South"
    elif nl.startswith("ew") or nl.startswith("we"):
        approach = "East-West"

    movement = ""
    if "thru" in nl or "through" in nl:
        movement = "main green band for through (and usually right-turn) lanes on"
    elif "left" in nl:
        movement = "protected left-turn green on"
    else:
        movement = "green / movement pattern on"

    if approach:
        core = f"{movement} {approach} approaches (as wired in your SUMO network); other legs follow red/yellow per the state string."
    else:
        core = f"«{raw.replace('_', ' ')}»: green/yellow/red pattern per link in sumo_state."

    return core + " " + _link_signal_summary(state)


def _enrich_legend_row(index: int, name: str, sumo_state: str) -> dict[str, Any]:
    title = (name or "").strip().replace("_", " ") or f"phase {index}"
    return {
        "index": index,
        "name": name or "",
        "title": title,
        "meaning": _human_phase_meaning(name, sumo_state),
        "sumo_state": sumo_state,
    }


def parse_tll_phase_legend(tll_path: str) -> dict[str, list[dict[str, Any]]]:
    """
    Load ``*.tll.xml``: for each ``tlLogic``, ordered list of phases with
    index, SUMO name/state, and human-readable fields.
    """
    tree = ET.parse(tll_path)
    root = tree.getroot()
    out: dict[str, list[dict[str, Any]]] = {}
    for tl in root.findall("tlLogic"):
        tls_id = tl.get("id") or ""
        rows: list[dict[str, Any]] = []
        for i, ph in enumerate(tl.findall("phase")):
            name = ph.get("name") or ""
            st = ph.get("state") or ""
            row = _enrich_legend_row(i, name, st)
            dur = ph.get("duration")
            if dur is not None:
                try:
                    row["program_duration_s_in_file"] = float(dur)
                except ValueError:
                    row["program_duration_s_in_file"] = dur
            rows.append(row)
        if tls_id:
            out[tls_id] = rows
    return out


def build_phase_legend_from_test_sequences(output_dir: str) -> dict[str, list[dict[str, Any]]] | None:
    """
    If ``*.tll.xml`` is missing, infer phase names/states from exported
    ``test_sequences/episode_*.json`` (last non-empty name/state wins per phase).
    """
    seq_dir = Path(output_dir) / "test_sequences"
    if not seq_dir.is_dir():
        return None
    by_tls: dict[str, dict[int, dict[str, str]]] = {}
    for path in sorted(seq_dir.glob("episode_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        seq = data.get("tls_sequences") or {}
        for tls_id, events in seq.items():
            bucket = by_tls.setdefault(tls_id, {})
            for ev in events:
                try:
                    idx = int(ev["phase"])
                except (TypeError, KeyError, ValueError):
                    continue
                rec = bucket.setdefault(idx, {"name": "", "state": ""})
                nm = (ev.get("phase_name") or "").strip()
                st = (ev.get("phase_state") or "").strip()
                if nm:
                    rec["name"] = nm
                if st:
                    rec["state"] = st
    if not by_tls:
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    for tls_id, phases in by_tls.items():
        rows = [
            _enrich_legend_row(i, phases[i]["name"], phases[i]["state"])
            for i in sorted(phases.keys())
        ]
        out[tls_id] = rows
    return out


def resolve_phase_legend(
    *,
    tll_path: str | None = None,
    net_file: str | None = None,
    output_dir: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]] | None, str | None]:
    """
    Return (legend dict, resolved tll path used) or (None, None).
    Prefers ``tll_path``, then infers from ``net_file``, then test_sequences.
    """
    path = tll_path or infer_tll_path_from_net(net_file)
    if path:
        try:
            leg = parse_tll_phase_legend(path)
            if leg:
                return leg, path
        except (ET.ParseError, OSError):
            pass
    if output_dir:
        leg = build_phase_legend_from_test_sequences(output_dir)
        if leg:
            return leg, None
    return None, None


def enrich_buckets_with_legend(
    buckets: list[dict[str, Any]],
    legend: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Add ``phase_title``, ``phase_meaning``, ``sumo_state`` to each bucket row."""
    by_tls_phase: dict[tuple[str, int], dict[str, Any]] = {}
    for tls_id, rows in legend.items():
        for row in rows:
            by_tls_phase[(tls_id, int(row["index"]))] = row

    enriched: list[dict[str, Any]] = []
    for b in buckets:
        tls_id = b["tls_id"]
        ph = int(b["phase"])
        meta = by_tls_phase.get((tls_id, ph))
        nb = dict(b)
        if meta:
            nb["phase_title"] = meta.get("title", f"phase {ph}")
            nb["phase_meaning"] = meta.get("meaning", "")
            nb["sumo_state"] = meta.get("sumo_state", "")
        else:
            nb["phase_title"] = f"phase {ph} (not in legend)"
            nb["phase_meaning"] = (
                "No matching entry in phase_legend — check that your .tll.xml matches this run."
            )
            nb["sumo_state"] = ""
        enriched.append(nb)
    return enriched


def build_schedule_document(
    buckets: list[dict[str, Any]],
    *,
    tll_path: str | None = None,
    net_file: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    legend, resolved_tll = resolve_phase_legend(
        tll_path=tll_path, net_file=net_file, output_dir=output_dir
    )
    about = {
        "buckets": (
            "Per 15-minute simulation-time window: how long (median_s) the "
            "controller stayed in that SUMO phase while the vehicle-demand "
            "window (day CSV) covered sim_time. n = number of segment samples."
        ),
        "phase_legend": (
            "Each phase index is the order of <phase> elements in SUMO tlLogic "
            "program 0. sumo_state: one character per traffic-light link in SUMO "
            "internal link order (G=major green, g=minor green, y=yellow, r=red)."
        ),
    }
    doc: dict[str, Any] = {"_about": about}
    if resolved_tll:
        doc["tll_source"] = Path(resolved_tll).as_posix()
    elif legend and output_dir:
        doc["tll_source"] = None
        doc["phase_legend_source"] = "test_sequences (tll not found or unreadable)"
    if legend:
        doc["phase_legend"] = legend
        doc["buckets"] = enrich_buckets_with_legend(buckets, legend)
    else:
        doc["buckets"] = buckets
    return doc


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


def write_schedule_json(
    path: str,
    buckets: list[dict[str, Any]],
    *,
    tll_path: str | None = None,
    net_file: str | None = None,
    output_dir: str | None = None,
) -> None:
    doc = build_schedule_document(
        buckets,
        tll_path=tll_path,
        net_file=net_file,
        output_dir=output_dir or str(Path(path).parent),
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


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
    *,
    net_file: str | None = None,
    tll_path: str | None = None,
) -> str | None:
    """Aggregate and write ``schedule.json`` under ``output_dir``; return path or None."""
    entries = collect_schedule_entries_from_run(output_dir, test_log, days_dir, warn=warn)
    if not entries:
        return None
    schedule = aggregate_schedule(entries)
    path = str(Path(output_dir) / "schedule.json")
    write_schedule_json(
        path,
        schedule,
        tll_path=tll_path,
        net_file=net_file,
        output_dir=output_dir,
    )
    return path
