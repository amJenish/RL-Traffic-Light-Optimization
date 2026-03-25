"""
Helpers to suggest intersection.json from an uploaded CSV + optional columns.json.

Traffic counts alone do not define lane geometry; we infer active approaches from
column names / column map, then apply user-supplied or default lane counts and
standard timing defaults (editable in the UI).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

# Defaults aligned with src/intersection.json — adjust in one place.
DEFAULT_TIMING: dict[str, Any] = {
    "phases": "4",
    "min_red_s": 15,
    "min_green_s": 15,
    "max_green_s": 90,
    "amber_s": 3,
    "cycle_s": 120,
    "edge_length_m": 200,
}

APPROACH_ORDER = ["N", "S", "E", "W"]


def _slug_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    s = s.strip("_")
    return s or "Intersection"


def detect_approaches_from_headers(df: pd.DataFrame) -> list[str]:
    """
    Heuristic: columns like n_approaching_t, s_approaching_r, etc.
    """
    found: set[str] = set()
    for col in df.columns:
        cl = str(col).lower()
        m = re.match(r"^([nsew])_approaching", cl)
        if m:
            found.add(m.group(1).upper())
    return [d for d in APPROACH_ORDER if d in found]


def detect_approaches_from_column_map(df: pd.DataFrame, col_map: dict) -> tuple[list[str], list[str]]:
    """
    Use columns.json approaches: active if any mapped column name exists in df.
    Returns (active_approaches, warnings).
    """
    warnings: list[str] = []
    active: list[str] = []
    approaches = col_map.get("approaches") or {}
    for direction in APPROACH_ORDER:
        cfg = approaches.get(direction)
        if not cfg:
            continue
        any_mapped = False
        for key in ("through", "right", "left", "peds"):
            mapped = cfg.get(key)
            if mapped is None:
                continue
            col_set = {str(c) for c in df.columns}
            if isinstance(mapped, str) and mapped in col_set:
                any_mapped = True
        if any_mapped:
            active.append(direction)
        else:
            if any(cfg.get(k) for k in ("through", "right", "left", "peds") if cfg.get(k)):
                warnings.append(
                    f"Approach {direction} is mapped in columns.json but no mapped column "
                    f"names were found in the CSV."
                )
    return active, warnings


def validate_csv_minimum(df: pd.DataFrame, col_map: dict | None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for pipeline readiness."""
    errors: list[str] = []
    warnings: list[str] = []
    if df is None or df.empty:
        errors.append("CSV is empty.")
        return errors, warnings

    if col_map:
        time_cfg = col_map.get("time") or {}
        start = time_cfg.get("start_time")
        if start and isinstance(start, str):
            if start not in df.columns:
                errors.append(
                    f"Column map requires start_time column '{start}' but it is not in the CSV."
                )
        elif time_cfg.get("start_time"):
            errors.append("time.start_time in columns.json must be a string column name or null.")

    if col_map is None:
        if "start_time" not in df.columns:
            warnings.append(
                "No columns.json parsed: if your CSV uses another name for the time column, "
                "add a column map or rename to start_time after mapping."
            )

    return errors, warnings


def suggest_intersection_name(df: pd.DataFrame) -> str:
    for cand in ("location_name", "Location", "intersection"):
        if cand in df.columns:
            v = df[cand].dropna().astype(str).unique()
            if len(v) > 0:
                return _slug_name(str(v[0]))
    return "My_Intersection"


def build_intersection_dict(
    intersection_name: str,
    active_approaches: list[str],
    lanes_through: int,
    lanes_right: int,
    lanes_left: int,
    speed_kmh: float,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing = {**DEFAULT_TIMING, **(timing or {})}
    approaches: dict[str, Any] = {}
    for d in active_approaches:
        approaches[d] = {
            "lanes": {
                "through": int(lanes_through),
                "right": int(lanes_right),
                "left": int(lanes_left),
            },
            "speed_kmh": float(speed_kmh),
        }
    return {
        "intersection_name": _slug_name(intersection_name),
        "active_approaches": list(active_approaches),
        "approaches": approaches,
        **timing,
    }


def _pick_col(df: pd.DataFrame, prefix: str, suffix: str) -> str | None:
    key = f"{prefix}_approaching_{suffix}"
    for c in df.columns:
        if str(c).lower() == key:
            return str(c)
    return None


def suggest_columns_json_from_dataframe(df: pd.DataFrame, active: list[str]) -> dict[str, Any]:
    """
    Build a minimal columns.json if headers follow the Toronto-style names.
    Maps null for missing optional columns.
    """
    cols = {str(c) for c in df.columns}
    time_block: dict[str, Any | None] = {
        "start_time": "start_time" if "start_time" in cols else None,
        "end_time": "end_time" if "end_time" in cols else None,
        "date": "date" if "date" in cols else None,
        "day_of_week": "day_of_week" if "day_of_week" in cols else None,
    }
    if not time_block["start_time"]:
        for c in df.columns:
            if "time" in str(c).lower():
                time_block["start_time"] = str(c)
                break

    approaches: dict[str, Any] = {}
    for d in active:
        p = d.lower()
        approaches[d] = {
            "through": _pick_col(df, p, "t"),
            "right": _pick_col(df, p, "r"),
            "left": _pick_col(df, p, "l"),
            "peds": _pick_col(df, p, "peds"),
        }

    return {
        "_instructions": "Auto-generated from CSV headers. Edit to match your file.",
        "time": time_block,
        "approaches": approaches,
        "slot_minutes": 15,
    }
