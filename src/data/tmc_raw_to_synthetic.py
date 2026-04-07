"""
Convert Toronto Open Data–style TMC CSV (e.g. tmc_raw_data_2020_2029.csv) into the
narrow `synthetic_toronto_data.csv` schema used by this repo:

  location_name, start_time, end_time, date, day_of_week,
  n/s/e/w_approaching_{t,r,l,peds}

Vehicle counts (cars + trucks + buses) are summed per movement; approach-level
bicycle counts are folded into the through bucket to match the 4 movement columns
expected downstream.

Modes
-----
direct     — one output row per real count day / time bin (after optional filters).
             Dates and day_of_week come from `count_date`.

synthetic  — for each location, take the earliest day in the filtered data as a
             15‑minute profile template, then emit `synthetic_days` calendar days
             starting at `synthetic_start_date`, applying multiplicative "nudge"
             noise to all *_approaching_* integer columns.

Usage (from repo root)
----------------------
  python src/data/tmc_raw_to_synthetic.py --help
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DIRS = ("n", "s", "e", "w")
MOVEMENT_SUFFIXES = ("t", "r", "l", "peds")

OUTPUT_COLUMNS = [
    "location_name",
    "start_time",
    "end_time",
    "date",
    "day_of_week",
    "n_approaching_t",
    "n_approaching_r",
    "n_approaching_l",
    "n_approaching_peds",
    "s_approaching_t",
    "s_approaching_r",
    "s_approaching_l",
    "s_approaching_peds",
    "e_approaching_t",
    "e_approaching_r",
    "e_approaching_l",
    "e_approaching_peds",
    "w_approaching_t",
    "w_approaching_r",
    "w_approaching_l",
    "w_approaching_peds",
]

COUNT_COLUMNS = [f"{d}_approaching_{s}" for d in DIRS for s in MOVEMENT_SUFFIXES]


def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int64)


def raw_to_approaching(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregated *_{t,r,l,peds} columns; keep originals."""
    out = df.copy()
    for d in DIRS:
        c_t = _coerce_int(out[f"{d}_appr_cars_t"]) + _coerce_int(out[f"{d}_appr_truck_t"])
        c_t += _coerce_int(out[f"{d}_appr_bus_t"]) + _coerce_int(out[f"{d}_appr_bike"])
        c_r = (
            _coerce_int(out[f"{d}_appr_cars_r"])
            + _coerce_int(out[f"{d}_appr_truck_r"])
            + _coerce_int(out[f"{d}_appr_bus_r"])
        )
        c_l = (
            _coerce_int(out[f"{d}_appr_cars_l"])
            + _coerce_int(out[f"{d}_appr_truck_l"])
            + _coerce_int(out[f"{d}_appr_bus_l"])
        )
        p = _coerce_int(out[f"{d}_appr_peds"])
        out[f"{d}_approaching_t"] = c_t
        out[f"{d}_approaching_r"] = c_r
        out[f"{d}_approaching_l"] = c_l
        out[f"{d}_approaching_peds"] = p
    return out


def normalize_times_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    st = pd.to_datetime(out["start_time"], errors="coerce")
    et = pd.to_datetime(out["end_time"], errors="coerce")
    out["start_time"] = st.dt.strftime("%H:%M:%S")
    out["end_time"] = et.dt.strftime("%H:%M:%S")
    out["count_date"] = pd.to_datetime(out["count_date"], errors="coerce").dt.normalize()
    return out


def group_count_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate keys (same location, calendar day, interval) by summing counts."""
    keys = ["location_name", "count_date", "start_time", "end_time"]
    return df.groupby(keys, as_index=False)[list(COUNT_COLUMNS)].sum()


def attach_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["count_date"].dt.strftime("%Y-%m-%d")
    out["day_of_week"] = out["count_date"].dt.dayofweek.astype(int)
    return out


def nudge_counts(
    df: pd.DataFrame,
    columns: Iterable[str],
    relative_span: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    span = float(relative_span)
    if span <= 0:
        return out
    for col in columns:
        base = out[col].to_numpy(dtype=np.float64)
        noise = rng.uniform(low=-span, high=span, size=len(out))
        out[col] = np.maximum(0, np.round(base * (1.0 + noise))).astype(np.int64)
    return out


def expand_synthetic_profiles(
    df: pd.DataFrame,
    synthetic_days: int,
    start: date,
    relative_nudge: float,
    seed: int,
) -> pd.DataFrame:
    """
    For each location, use min(count_date) rows as the daily template; emit
    `synthetic_days` copies with calendar dates from `start` and nudged counts.
    """
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for _, loc_df in df.groupby("location_name", sort=False):
        ref = loc_df["count_date"].min()
        if pd.isna(ref):
            continue
        tmpl = loc_df[loc_df["count_date"] == ref].copy()
        if tmpl.empty:
            continue
        tmpl = tmpl.drop(columns=["count_date"])
        for day_i in range(int(synthetic_days)):
            d = start + timedelta(days=day_i)
            block = tmpl.copy()
            block = nudge_counts(block, COUNT_COLUMNS, relative_nudge, rng)
            block["date"] = d.isoformat()
            block["day_of_week"] = int(d.weekday())
            blocks.append(block)
    if not blocks:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(blocks, ignore_index=True)


def load_filtered_raw(
    path: Path,
    locations: list[str] | None,
    date_from: str | None,
    date_to: str | None,
    chunksize: int | None,
) -> pd.DataFrame:
    usecols = None  # full row; keeps memory simpler for filter by location
    read_kw: dict = {"dtype": {"location_name": str}}
    if chunksize:
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, **read_kw):
            chunk = _apply_filters(chunk, locations, date_from, date_to)
            if not chunk.empty:
                parts.append(chunk)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)
    df = pd.read_csv(path, low_memory=False, **read_kw)
    return _apply_filters(df, locations, date_from, date_to)


def _apply_filters(
    df: pd.DataFrame,
    locations: list[str] | None,
    date_from: str | None,
    date_to: str | None,
) -> pd.DataFrame:
    out = df
    if locations:
        want = set(locations)
        out = out[out["location_name"].isin(want)]
    if date_from:
        t0 = pd.to_datetime(date_from)
        out = out[pd.to_datetime(out["count_date"], errors="coerce") >= t0]
    if date_to:
        t1 = pd.to_datetime(date_to)
        out = out[pd.to_datetime(out["count_date"], errors="coerce") <= t1]
    return out


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(path)

    df = load_filtered_raw(
        path,
        locations=list(args.location) if args.location else None,
        date_from=args.date_from,
        date_to=args.date_to,
        chunksize=args.chunksize,
    )
    if df.empty:
        raise ValueError("No rows left after filters.")

    df = raw_to_approaching(df)
    df = normalize_times_and_dates(df)
    df = df.dropna(subset=["count_date", "start_time", "end_time"])
    df = group_count_intervals(df)

    if args.mode == "direct":
        out = attach_calendar_columns(df)
        out = out.drop(columns=["count_date"])
    else:
        start = date.fromisoformat(args.synthetic_start_date)
        out = expand_synthetic_profiles(
            df,
            synthetic_days=args.synthetic_days,
            start=start,
            relative_nudge=args.nudge,
            seed=args.seed,
        )

    for c in OUTPUT_COLUMNS:
        if c not in out.columns:
            out[c] = 0 if c != "location_name" else ""

    out = out[OUTPUT_COLUMNS].sort_values(
        ["location_name", "date", "start_time"], kind="mergesort"
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=str,
        default="tmc_raw_data_2020_2029.csv",
        help="Path to raw TMC CSV.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="src/data/synthetic_toronto_from_tmc.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--mode",
        choices=("direct", "synthetic"),
        default="direct",
        help="direct: real dates; synthetic: template + nudged multi-day series.",
    )
    p.add_argument(
        "--location",
        action="append",
        dest="location",
        default=[],
        help="Restrict to a location_name (repeatable).",
    )
    p.add_argument("--date-from", type=str, default=None, help="Inclusive YYYY-MM-DD.")
    p.add_argument("--date-to", type=str, default=None, help="Inclusive YYYY-MM-DD.")
    p.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Rows per chunk when reading CSV (saves RAM on huge files).",
    )
    p.add_argument(
        "--synthetic-days",
        type=int,
        default=7,
        help="Number of consecutive calendar days in synthetic mode.",
    )
    p.add_argument(
        "--synthetic-start-date",
        type=str,
        default="2024-01-01",
        help="First calendar date in synthetic mode (YYYY-MM-DD).",
    )
    p.add_argument(
        "--nudge",
        type=float,
        default=0.12,
        help="Uniform multiplicative noise in [-nudge, +nudge] per count column.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for synthetic mode.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run_pipeline(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
