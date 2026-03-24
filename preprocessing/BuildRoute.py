"""
preprocess.py
-------------
Converts any traffic count CSV into per-day files and SUMO flow files,
driven entirely by two config files — no hardcoded column names or edge IDs.

Merged from SUMO-Demand-Generation-Pipeline/data_to_rou.py:
  - normalize_col()  — tolerates any case/spacing in CSV headers
  - --date-mode      — error | offset | concat (multi-date handling)
  - --write-sumocfg  — write a .sumocfg for sumo-gui visual inspection

Required inputs:
    --csv              Path to traffic count CSV (or .xls / .xlsx)
    --intersection     Path to intersection_config.json
    --columns          Path to column_map.json  (maps CSV headers)

Outputs:
    data/processed/days/day_NN.csv         One clean CSV per simulated day
    data/processed/split.json              Train / test split + shared config
    data/sumo/flows/flows_day_NN.rou.xml   SUMO demand file per day
    simulation.sumocfg                     (optional, --write-sumocfg)

Usage:
    python preprocess.py \
        --csv            synthetic_toronto_data.xls \
        --intersection   intersection_config.json \
        --columns        column_map.json

    python preprocess.py \
        --csv            synthetic_toronto_data.xls \
        --intersection   intersection_config.json \
        --columns        column_map.json \
        --date-mode      concat \
        --write-sumocfg
"""

import argparse
import json
import math
import os
import re
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

import pandas as pd


OVERNIGHT_VPH = 12
SIM_DURATION  = 86400

# Exit approach for each (from_approach, movement) pair.
# Convention: right/left are from the driver's perspective.
#   N approach → traveling South:  right=W, left=E
#   S approach → traveling North:  right=E, left=W
#   E approach → traveling West:   right=N, left=S
#   W approach → traveling East:   right=S, left=N
TURN_TARGETS = {
    "N": {"through": "S", "right": "W", "left": "E"},
    "S": {"through": "N", "right": "E", "left": "W"},
    "E": {"through": "W", "right": "N", "left": "S"},
    "W": {"through": "E", "right": "S", "left": "N"},
}


# ---------------------------------------------------------------------------
# COLUMN NORMALISATION  (from data_to_rou.py)
# ---------------------------------------------------------------------------

def normalize_col(s: str) -> str:
    """Lowercase and collapse any runs of whitespace / underscores to '_'.

    This makes column matching case- and spacing-insensitive, so
    'N Approaching T', 'n_approaching_t', and 'N_Approaching_T' all
    become 'n_approaching_t'.
    """
    return re.sub(r"[\s_]+", "_", s.lower().strip())


# ---------------------------------------------------------------------------
# INTERSECTION / COLUMN MAP LOADING
# ---------------------------------------------------------------------------

def load_intersection(path):
    with open(path) as f:
        cfg = json.load(f)
    for k in ["intersection_name", "active_approaches", "approaches"]:
        if k not in cfg:
            raise ValueError(f"intersection_config.json missing key: '{k}'")
    return cfg


def load_column_map(path):
    with open(path) as f:
        return json.load(f)


def edge_in_id(approach):
    return f"edge_{approach}_in"


def edge_out_id(approach):
    return f"edge_{approach}_out"


# Default through-movement destination for each approach
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


# ---------------------------------------------------------------------------
# CSV LOADING  (enhanced: normalizes column names before matching)
# ---------------------------------------------------------------------------

def load_csv(csv_path, col_map, active_approaches):
    # Support .xls / .xlsx as well as .csv
    suffix = os.path.splitext(csv_path)[1].lower()
    if suffix in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(csv_path)
        except Exception:
            df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Normalize every column name so matching is case/spacing insensitive
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    junk = [c for c in df.columns if c.startswith("unnamed")]
    df = df.drop(columns=junk)

    time_cfg  = col_map.get("time", {})
    start_col = normalize_col(time_cfg.get("start_time") or "") if time_cfg.get("start_time") else None
    end_col   = normalize_col(time_cfg.get("end_time")   or "") if time_cfg.get("end_time")   else None
    date_col  = normalize_col(time_cfg.get("date")       or "") if time_cfg.get("date")       else None
    dow_col   = normalize_col(time_cfg.get("day_of_week") or "") if time_cfg.get("day_of_week") else None

    if not start_col or start_col not in df.columns:
        raise ValueError(
            f"start_time column '{start_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Fix: column_map.json -> time -> start_time"
        )

    rename = {start_col: "start_time"}
    if end_col and end_col in df.columns:
        rename[end_col] = "end_time"
    if date_col and date_col in df.columns:
        rename[date_col] = "date"
    if dow_col and dow_col in df.columns:
        rename[dow_col] = "day_of_week"
    df = df.rename(columns=rename)

    approach_cfg = col_map.get("approaches", {})
    for d in active_approaches:
        if d not in approach_cfg:
            raise ValueError(
                f"Approach '{d}' active in intersection_config but not in column_map.json"
            )
        for movement in ["through", "right", "left"]:
            src      = approach_cfg[d].get(movement)
            src_norm = normalize_col(src) if src else None
            dst      = f"{d}_{movement}"
            if src_norm and src_norm in df.columns:
                df[dst] = df[src_norm]
            else:
                if src:
                    print(f"  Warning: column '{src}' not found for {d}/{movement} — using 0")
                df[dst] = 0

    for d in active_approaches:
        df[f"{d}_total"] = (
            df[f"{d}_through"] + df[f"{d}_right"] + df[f"{d}_left"]
        )

    ped_cols = [c for c in df.columns if "ped" in c.lower()]
    df = df.drop(columns=ped_cols)
    return df


# ---------------------------------------------------------------------------
# DAY ASSIGNMENT  (enhanced: supports date-mode from data_to_rou.py)
# ---------------------------------------------------------------------------

def assign_sim_days(df, date_mode: str = "concat"):
    """Assign a sim_day integer to every row.

    date_mode:
      concat  (default) — Sort unique dates ascending; assign day 0, 1, …
                          in that order. Matches original behaviour.
      offset  — Use (date - first_date).days as the day index so calendar
                spacing is preserved (e.g. a gap leaves unused day indices).
      error   — Raise ValueError if more than one distinct date is present.
    """
    df = df.copy()

    if "date" in df.columns:
        dates = sorted(df["date"].dropna().unique())
    else:
        dates = []

    n_dates = len(dates)

    if date_mode == "error" and n_dates > 1:
        raise ValueError(
            f"Multiple distinct dates found ({n_dates} days). "
            f"Use --date-mode concat or offset, or restrict input to a single day."
        )

    if date_mode == "offset" and n_dates > 1:
        first = dates[0]
        try:
            df["sim_day"] = df["date"].apply(
                lambda d: (pd.Timestamp(str(d)) - pd.Timestamp(str(first))).days
            )
        except Exception:
            # Fallback to positional cumcount if timestamp parse fails
            df["sim_day"] = df.groupby("start_time").cumcount()

    elif n_dates > 1:
        # concat (default): ascending date → ascending day index
        date_to_id = {d: i for i, d in enumerate(dates)}
        df["sim_day"] = df["date"].map(date_to_id)

    else:
        # Single date or no date column — positional cumcount within each slot
        df["sim_day"] = df.groupby("start_time").cumcount()

    df = df.sort_values(["sim_day", "start_time"]).reset_index(drop=True)
    return df


def split_days(n_days, n_test, seed=42):
    rng = random.Random(seed)
    all_days   = list(range(n_days))
    test_days  = sorted(rng.sample(all_days, n_test))
    train_days = sorted(set(all_days) - set(test_days))
    return {"train": train_days, "test": test_days}


# ---------------------------------------------------------------------------
# OUTPUT WRITERS
# ---------------------------------------------------------------------------

def write_day_csvs(df, active, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    movement_cols = []
    for d in active:
        for m in ["through", "right", "left"]:
            movement_cols.append(f"{d}_{m}")
        movement_cols.append(f"{d}_total")

    base_cols = ["sim_day", "start_time"]
    if "end_time"    in df.columns: base_cols.append("end_time")
    if "date"        in df.columns: base_cols.append("date")
    if "day_of_week" in df.columns: base_cols.append("day_of_week")
    keep = base_cols + movement_cols

    n_days = df["sim_day"].nunique()
    for day_id in range(n_days):
        day_df = df[df["sim_day"] == day_id][keep].reset_index(drop=True)
        day_df.to_csv(os.path.join(out_dir, f"day_{day_id:02d}.csv"), index=False)
    print(f"  Written {n_days} day CSV files -> {out_dir}")
    return n_days


def time_to_seconds(t):
    parts = str(t).strip().split(":")
    h, m = int(parts[0]), int(parts[1])
    s    = int(parts[2]) if len(parts) > 2 else 0
    return h * 3600 + m * 60 + s


def _allowed_movements(active, int_cfg):
    """Return a set of (approach, movement) tuples that have physical lanes
    AND whose turn target is an active approach (handles T-intersections)."""
    allowed = set()
    approaches = int_cfg.get("approaches", {})
    for d in active:
        lanes = approaches.get(d, {}).get("lanes", {})
        for movement in ("through", "right", "left"):
            if lanes.get(movement, 0) == 0:
                continue
            target = TURN_TARGETS.get(d, {}).get(movement)
            if target and target in active:
                allowed.add((d, movement))
    return allowed


def write_sumo_flows(df, active, slot_minutes, out_dir, int_cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    slots_sorted = sorted(df["start_time"].unique())
    data_start   = time_to_seconds(slots_sorted[0])
    data_end     = time_to_seconds(slots_sorted[-1]) + slot_minutes * 60
    vph_factor   = 60.0 / slot_minutes
    n_days       = df["sim_day"].nunique()

    if int_cfg is None:
        int_cfg = {}

    allowed = _allowed_movements(active, int_cfg)

    max_speed_kmh = max(
        (int_cfg.get("approaches", {}).get(d, {}).get("speed_kmh", 50)
         for d in active),
        default=50,
    )
    max_speed_ms = max_speed_kmh / 3.6

    for day_id in range(n_days):
        day_df = df[df["sim_day"] == day_id].sort_values("start_time")
        root   = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation",
                 "http://sumo.dlr.de/xsd/routes_file.xsd")

        vt = ET.SubElement(root, "vType")
        vt.set("id","passenger"); vt.set("accel","2.6"); vt.set("decel","4.5")
        vt.set("sigma","0.5");    vt.set("length","5")
        vt.set("maxSpeed", f"{max_speed_ms:.2f}")

        fid = [0]

        def add_flow(begin, end, from_approach, to_approach, vph):
            if vph < 0.5: return
            f = ET.SubElement(root, "flow")
            f.set("id",          f"f{fid[0]}")
            f.set("type",        "passenger")
            f.set("from",        edge_in_id(from_approach))
            f.set("to",          edge_out_id(to_approach))
            f.set("begin",       str(int(begin)))
            f.set("end",         str(int(end)))
            f.set("vehsPerHour", f"{vph:.1f}")
            f.set("departLane",  "best")
            f.set("departSpeed", "max")
            fid[0] += 1

        n_allowed = max(len(allowed), 1)

        if data_start > 0:
            for d, movement in sorted(allowed):
                target = TURN_TARGETS[d][movement]
                add_flow(0, data_start, d, target, OVERNIGHT_VPH / n_allowed)

        for _, row in day_df.iterrows():
            begin = time_to_seconds(row["start_time"])
            end   = begin + slot_minutes * 60
            for d, movement in sorted(allowed):
                target = TURN_TARGETS[d][movement]
                add_flow(begin, end, d, target,
                         row[f"{d}_{movement}"] * vph_factor)

        remaining = SIM_DURATION - data_end
        if remaining > 0:
            last_row = day_df.iloc[-1]
            n_taper  = math.ceil(remaining / (slot_minutes * 60))
            for i in range(n_taper):
                t_frac = i / max(n_taper - 1, 1)
                begin  = data_end + i * slot_minutes * 60
                end    = min(begin + slot_minutes * 60, SIM_DURATION)
                for d, movement in sorted(allowed):
                    target = TURN_TARGETS[d][movement]
                    last_vph = last_row[f"{d}_{movement}"] * vph_factor
                    add_flow(begin, end, d, target,
                             last_vph * (1 - t_frac)
                             + (OVERNIGHT_VPH / n_allowed) * t_frac)

        raw   = ET.tostring(root, encoding="unicode")
        lines = minidom.parseString(raw).toprettyxml(indent="  ").split("\n")
        lines = [l for l in lines if not l.startswith("<?xml")]
        path  = os.path.join(out_dir, f"flows_day_{day_id:02d}.rou.xml")
        with open(path, "w") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write("\n".join(lines))

    print(f"  Written {n_days} SUMO flow files -> {out_dir}")


def write_sumocfg(net_file: str, flows_dir: str, out_path: str,
                  begin: int = 27000, end: int = 64800) -> None:
    """Write a .sumocfg for visual inspection with sumo-gui.

    References the built network and all flow files in flows_dir as a
    comma-separated list so any single day can be loaded via --route-files.
    Ported from SUMO-Demand-Generation-Pipeline/data_to_rou.py.
    """
    route_files = sorted(
        f for f in os.listdir(flows_dir) if f.endswith(".rou.xml")
    ) if os.path.isdir(flows_dir) else []
    route_str = ",".join(route_files) if route_files else "flows_day_00.rou.xml"

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
        f.write('  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
        f.write('  <input>\n')
        f.write(f'    <net-file value="{os.path.abspath(net_file)}"/>\n')
        f.write(f'    <route-files value="{route_str}"/>\n')
        f.write('  </input>\n')
        f.write('  <time>\n')
        f.write(f'    <begin value="{begin}"/>\n')
        f.write(f'    <end value="{end}"/>\n')
        f.write('  </time>\n')
        f.write('</configuration>\n')

    print(f"  sumocfg written -> {out_path}")
    print(f"  Run: sumo-gui -c {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess traffic CSV for SUMO + RL")
    parser.add_argument("--csv",          required=True,
                        help="Path to traffic count CSV / XLS / XLSX")
    parser.add_argument("--intersection", required=True,
                        help="Path to intersection_config.json")
    parser.add_argument("--columns",      required=True,
                        help="Path to column_map.json")
    parser.add_argument("--out-dir",      default="data",
                        help="Root output directory (default: data)")
    parser.add_argument("--test-days",    type=int, default=6,
                        help="Number of days held out for testing (default: 6)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed for train/test split (default: 42)")
    parser.add_argument(
        "--date-mode",
        choices=("error", "offset", "concat"),
        default="concat",
        help=(
            "How to handle multiple dates in the CSV  "
            "(from SUMO-Demand-Generation-Pipeline/data_to_rou.py):\n"
            "  concat  — Stack days 0, 1, … by ascending date order [default]\n"
            "  offset  — Use (date − first_date).days as the day index\n"
            "  error   — Fail if more than one distinct date is present"
        ),
    )
    parser.add_argument(
        "--write-sumocfg",
        action="store_true",
        help=(
            "Write a simulation.sumocfg for sumo-gui visual inspection  "
            "(from SUMO-Demand-Generation-Pipeline/data_to_rou.py). "
            "Requires BuildNetwork.py to have already built the .net.xml."
        ),
    )
    parser.add_argument(
        "--sumocfg-out",
        default="simulation.sumocfg",
        help="Output path for the .sumocfg file (default: simulation.sumocfg)",
    )
    args = parser.parse_args()

    print(f"Loading intersection config: {args.intersection}")
    int_cfg = load_intersection(args.intersection)
    active  = int_cfg["active_approaches"]
    name    = int_cfg["intersection_name"]
    min_red = int_cfg.get("min_red_s", 15)
    print(f"  Intersection : {name}")
    print(f"  Approaches   : {active}")

    print(f"Loading column map: {args.columns}")
    col_map      = load_column_map(args.columns)
    slot_minutes = col_map.get("slot_minutes", 15)
    print(f"  Slot duration: {slot_minutes} min")

    print(f"Loading CSV: {args.csv}")
    df = load_csv(args.csv, col_map, active)

    print(f"Assigning simulated day IDs (date-mode={args.date_mode}) …")
    df      = assign_sim_days(df, date_mode=args.date_mode)
    n_days  = df["sim_day"].nunique()
    n_slots = df["start_time"].nunique()
    print(f"  {n_days} simulated days  |  {n_slots} slots per day")

    if args.test_days >= n_days:
        raise ValueError(f"test_days ({args.test_days}) must be < total days ({n_days})")

    processed_dir = os.path.join(args.out_dir, "processed")
    days_dir      = os.path.join(processed_dir, "days")
    flows_dir     = os.path.join(args.out_dir, "sumo", "flows")

    print("Writing per-day CSVs …")
    write_day_csvs(df, active, days_dir)

    print("Writing SUMO flow files …")
    write_sumo_flows(df, active, slot_minutes, flows_dir, int_cfg)

    print("Creating train/test split …")
    split = split_days(n_days, args.test_days, args.seed)

    dow_map = {}
    if "day_of_week" in df.columns:
        for day_id in range(n_days):
            day_df = df[df["sim_day"] == day_id]
            if not day_df.empty:
                dow_map[str(day_id)] = int(day_df["day_of_week"].iloc[0])

    split.update({
        "intersection_name": name,
        "active_approaches": active,
        "min_red_seconds":   min_red,
        "n_days":            n_days,
        "n_slots":           n_slots,
        "slot_minutes":      slot_minutes,
        "day_of_week_map":   dow_map,
        "edge_ids": {
            d: {"in": edge_in_id(d), "out": edge_out_id(d)}
            for d in active
        }
    })

    split_path = os.path.join(processed_dir, "split.json")
    os.makedirs(processed_dir, exist_ok=True)
    with open(split_path, "w") as fh:
        json.dump(split, fh, indent=2)

    print(f"  Train: {len(split['train'])} days  |  Test: {len(split['test'])} days")
    print(f"  Saved -> {split_path}")
    print(f"\nEdge IDs written to flow files:")
    for d in active:
        print(f"  {d}: {edge_in_id(d)} -> {edge_out_id(OPPOSITE[d])}")

    if args.write_sumocfg:
        net_dir  = os.path.join(args.out_dir, "sumo", "network")
        net_file = os.path.join(net_dir, f"{name}.net.xml")
        if os.path.exists(net_file):
            write_sumocfg(net_file, flows_dir, args.sumocfg_out,
                          begin=27000, end=64800)
        else:
            print(
                f"  Warning: network file not found at {net_file} — skipping sumocfg.\n"
                f"  Run BuildNetwork.py first, then re-run with --write-sumocfg."
            )

    print(f"\nDone. Run the RL trainer with:")
    print(f"  python train.py --data {args.out_dir}/processed/split.json")


if __name__ == "__main__":
    main()
