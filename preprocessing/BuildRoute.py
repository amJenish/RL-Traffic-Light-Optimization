"""
BuildRoute.py — converts a traffic count CSV into per-day files and SUMO flow files,
driven by intersection.json and columns.json.
"""

import argparse
import hashlib
import json
import math
import os
import re
import random
import xml.etree.ElementTree as ET
from typing import Optional
from xml.dom import minidom

import pandas as pd


OVERNIGHT_VPH = 12
SIM_DURATION = 86400

TURN_TARGETS = {
    "N": {"through": "S", "right": "W", "left": "E"},
    "S": {"through": "N", "right": "E", "left": "W"},
    "E": {"through": "W", "right": "N", "left": "S"},
    "W": {"through": "E", "right": "S", "left": "N"},
}


def normalize_col(s: str) -> str:
    """Lowercase and collapse whitespace/underscores for case-insensitive column matching."""
    return re.sub(r"[\s_]+", "_", s.lower().strip())


def load_intersection(path):
    """Read intersection config JSON. Requires active_approaches and approaches.

    intersection_name is optional (used for network file basename and logging);
    if missing or blank, defaults to ``My_Intersection``.
    """
    with open(path) as f:
        cfg = json.load(f)
    for k in ["active_approaches", "approaches"]:
        if k not in cfg:
            raise ValueError(f"intersection_config.json missing key: '{k}'")
    raw = cfg.get("intersection_name")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        cfg["intersection_name"] = "My_Intersection"
    else:
        cfg["intersection_name"] = str(raw).strip()
    return cfg


def load_column_map(path):
    with open(path) as f:
        return json.load(f)


def edge_in_id(approach):
    return f"edge_{approach}_in"


def edge_out_id(approach):
    return f"edge_{approach}_out"


OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def load_csv(csv_path, col_map, active_approaches):
    """Load CSV/XLS, normalise columns, extract approach movements using the column map."""
    suffix = os.path.splitext(csv_path)[1].lower()
    if suffix in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(csv_path)
        except Exception:
            df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    junk = [c for c in df.columns if c.startswith("unnamed")]
    df = df.drop(columns=junk)

    time_cfg = col_map.get("time", {})
    start_col = (
        normalize_col(time_cfg.get("start_time") or "")
        if time_cfg.get("start_time")
        else None
    )
    end_col = (
        normalize_col(time_cfg.get("end_time") or "")
        if time_cfg.get("end_time")
        else None
    )
    date_col = (
        normalize_col(time_cfg.get("date") or "") if time_cfg.get("date") else None
    )
    dow_col = (
        normalize_col(time_cfg.get("day_of_week") or "")
        if time_cfg.get("day_of_week")
        else None
    )

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
            src = approach_cfg[d].get(movement)
            src_norm = normalize_col(src) if src else None
            dst = f"{d}_{movement}"
            if src_norm and src_norm in df.columns:
                df[dst] = df[src_norm]
            else:
                if src:
                    print(
                        f"  Warning: column '{src}' not found for {d}/{movement} — using 0"
                    )
                df[dst] = 0

    for d in active_approaches:
        df[f"{d}_total"] = df[f"{d}_through"] + df[f"{d}_right"] + df[f"{d}_left"]

    ped_cols = [c for c in df.columns if "ped" in c.lower()]
    df = df.drop(columns=ped_cols)
    return df


def assign_sim_days(df, date_mode: str = "concat"):
    """Assign a sim_day integer to every row based on date_mode (concat|offset|error)."""
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
            df["sim_day"] = df.groupby("start_time").cumcount()

    elif n_dates > 1:
        date_to_id = {d: i for i, d in enumerate(dates)}
        df["sim_day"] = df["date"].map(date_to_id)

    else:
        df["sim_day"] = df.groupby("start_time").cumcount()

    df = df.sort_values(["sim_day", "start_time"]).reset_index(drop=True)
    return df


def split_days(n_days, n_test, seed=42):
    """Randomly split day indices into train and test sets."""
    rng = random.Random(seed)
    all_days = list(range(n_days))
    test_days = sorted(rng.sample(all_days, n_test))
    train_days = sorted(set(all_days) - set(test_days))
    return {"train": train_days, "test": test_days}


def write_day_csvs(df, active, out_dir):
    """Write one CSV per sim_day with movement columns."""
    os.makedirs(out_dir, exist_ok=True)
    movement_cols = []
    for d in active:
        for m in ["through", "right", "left"]:
            movement_cols.append(f"{d}_{m}")
        movement_cols.append(f"{d}_total")

    base_cols = ["sim_day", "start_time"]
    if "end_time" in df.columns:
        base_cols.append("end_time")
    if "date" in df.columns:
        base_cols.append("date")
    if "day_of_week" in df.columns:
        base_cols.append("day_of_week")
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
    s = int(parts[2]) if len(parts) > 2 else 0
    return h * 3600 + m * 60 + s


def _allowed_movements(active, int_cfg):
    """(approach, movement) pairs that have physical lanes and a valid exit approach."""
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


def _variation_rng(seed: int, day_id: int, row_index: int, tag: str) -> random.Random:
    """Deterministic RNG for demand variation (stable across Python runs)."""
    payload = f"{seed}:{day_id}:{row_index}:{tag}".encode()
    digest = hashlib.sha256(payload).digest()
    s = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return random.Random(s)


def _bounded_weights(k: int, spread: float, rng: random.Random) -> list[float]:
    """Return ``k`` weights with mean 1.0 for use as sub-slot demand multipliers.

    Weights are drawn as independent bounded-uniform perturbations around 1.0,
    renormalised to preserve expected vehicle count, with a ratio cap of
    ``(1 + 2 * spread)``.

    Raw samples ``u_i`` are i.i.d. After clipping to ``[clip_low, clip_high]``, if
    ``max(u)/min(u)`` exceeds the cap, the vector is **range-compressed** in a
    single affine pass (map ``[u_min, u_max]`` to ``[u_min, u_min * max_ratio]``),
    which preserves ordering and enforces ``max/min <= max_ratio`` before the final
    mean renormalisation (which does not change max/min ratio).

    Weights may be lifted to respect a 0.5 veh/h multiplier floor when paired with
    low ``base_vph``; if the ratio cap or floor cannot both be satisfied, uniform
    weights are used. Finally, weights are shuffled so sub-interval order does not
    follow the compression ordering (rigid steps, no temporal ramp).
    """
    if k < 1:
        return []
    low = 1.0 - float(spread)
    high = 1.0 + float(spread)
    clip_low = max(0.1, 1.0 - float(spread))
    clip_high = 1.0 + float(spread)
    max_ratio = 1.0 + 2.0 * float(spread)

    u: list[float] = []
    for _ in range(k):
        x = rng.uniform(low, high)
        x = min(max(x, clip_low), clip_high)
        u.append(x)

    u_min = min(u)
    u_max = max(u)
    if u_min <= 0.0:
        return [1.0] * k
    if (u_max / u_min) <= max_ratio + 1e-15:
        mean_u = sum(u) / k
        if mean_u <= 0.0:
            return [1.0] * k
        w = [ui / mean_u for ui in u]
    else:
        span = u_max - u_min
        hi_target = u_min * max_ratio
        new_span = hi_target - u_min
        if span <= 0.0 or new_span <= 0.0:
            return [1.0] * k
        scale = new_span / span
        u2 = [u_min + (ui - u_min) * scale for ui in u]
        mean2 = sum(u2) / k
        if mean2 <= 0.0:
            return [1.0] * k
        w = [x / mean2 for x in u2]

    # SUMO emission floor: sub-rates use base_vph * w_i; need multipliers >= 0.5 when base is 1.
    if min(w) < 0.5 - 1e-12:
        w = [max(xi, 0.5) for xi in w]
        mw = sum(w) / k
        if mw <= 0.0:
            return [1.0] * k
        w = [xi / mw for xi in w]
        if min(w) < 0.5 - 1e-12 or (max(w) / min(w)) > max_ratio + 1e-9:
            return [1.0] * k

    if (max(w) / min(w)) > max_ratio + 1e-9:
        return [1.0] * k

    order = list(range(k))
    rng.shuffle(order)
    return [w[i] for i in order]


def verify_bounded_subslot_weights() -> None:
    """Print V1–V6 diagnostics for :func:`_bounded_weights` (standalone QA)."""

    def _pearson(a: list[float], b: list[float]) -> float:
        n = len(a)
        if n < 2:
            return 0.0
        ma = sum(a) / n
        mb = sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        dxa = math.sqrt(sum((x - ma) ** 2 for x in a))
        dxb = math.sqrt(sum((y - mb) ** 2 for y in b))
        if dxa * dxb < 1e-18:
            return 0.0
        return num / (dxa * dxb)

    k = 5
    spread = 0.85
    max_r = 1.0 + 2.0 * spread
    n_sets = 1000
    rng = random.Random(12345)

    low, high = 1.0 - spread, 1.0 + spread
    clip_lo, clip_hi = max(0.1, 1.0 - spread), 1.0 + spread
    raw0, raw1 = [], []
    rng_v1 = random.Random(12345)
    for _ in range(n_sets):
        x0 = min(max(rng_v1.uniform(low, high), clip_lo), clip_hi)
        x1 = min(max(rng_v1.uniform(low, high), clip_lo), clip_hi)
        raw0.append(x0)
        raw1.append(x1)
    r01 = abs(_pearson(raw0, raw1))
    print(f"V1 independence: r={r01:.4f} — {'PASS' if r01 < 0.05 else 'FAIL'}")

    max_dev_mean = 0.0
    max_ratio_obs = 0.0
    cvs: list[float] = []
    for _ in range(n_sets):
        w = _bounded_weights(k, spread, rng)
        m = sum(w) / k
        max_dev_mean = max(max_dev_mean, abs(m - 1.0))
        max_ratio_obs = max(max_ratio_obs, max(w) / min(w))
        mu = sum(w) / k
        var = sum((x - mu) ** 2 for x in w) / k
        cvs.append(math.sqrt(var) / mu if mu else 0.0)

    print(f"V2 mean=1.0: max_deviation={max_dev_mean:.2e} — {'PASS' if max_dev_mean < 1e-9 else 'FAIL'}")
    print(
        f"V3 ratio cap: max_observed={max_ratio_obs:.4f} — "
        f"{'PASS' if max_ratio_obs <= max_r + 1e-9 else 'FAIL'}"
    )
    mean_cv = sum(cvs) / len(cvs)
    print(f"V4 spread CV: mean_cv={mean_cv:.4f} — {'PASS' if mean_cv >= 0.10 else 'FAIL'}")

    rng2 = random.Random(999)
    seq: list[float] = []
    for _ in range(20):
        seq.extend(_bounded_weights(5, spread, rng2))
    r_lag1 = abs(_pearson(seq[:-1], seq[1:]))
    print(f"V5 autocorrelation lag-1: r={r_lag1:.4f} — {'PASS' if r_lag1 < 0.10 else 'FAIL'}")

    rng3 = random.Random(42)
    triggered = False
    for _ in range(20000):
        w = _bounded_weights(5, spread, rng3)
        if min(w) < 0.5 - 1e-12:
            triggered = True
            break
    print(
        f"V6 floor at base_vph=1.0: triggered={triggered} — "
        f"{'PASS' if not triggered else 'FAIL'} (expected: False)"
    )


def _emit_subslot_flows(
    add_flow,
    slot_begin: float,
    slot_end: float,
    from_approach: str,
    to_approach: str,
    base_vph: float,
    *,
    full_slot_seconds: float,
    subslots_per_slot: int,
    spread: float,
    rng: Optional[random.Random],
) -> None:
    """Split one CSV slot into shorter SUMO flows with fluctuating vehsPerHour.

    Weights are drawn as independent bounded-uniform perturbations around 1.0,
    renormalised to preserve expected vehicle count, with a ratio cap of
    ``(1 + 2 * spread)``. Sub-rates are ``base_vph * weight_i``. Falls back to
    one flat flow if variation is off or any sub-rate would drop below SUMO's
    practical emission threshold (0.5 vehs/hour).
    """
    duration = slot_end - slot_begin
    if duration <= 0:
        return
    k = int(subslots_per_slot)
    if (
        k <= 1
        or spread <= 0.0
        or rng is None
        or duration < full_slot_seconds - 1e-6
    ):
        add_flow(slot_begin, slot_end, from_approach, to_approach, base_vph)
        return

    weights = _bounded_weights(k, spread, rng)
    vph_list = [base_vph * wi for wi in weights]
    if min(vph_list) < 0.5:
        add_flow(slot_begin, slot_end, from_approach, to_approach, base_vph)
        return

    step = duration / k
    for i in range(k):
        b = slot_begin + i * step
        e = slot_end if i == k - 1 else slot_begin + (i + 1) * step
        add_flow(b, e, from_approach, to_approach, vph_list[i])


def write_sumo_flows(
    df,
    active,
    slot_minutes,
    out_dir,
    int_cfg=None,
    *,
    subslots_per_slot: int = 5,
    spread: float = 0.85,
    variation_seed: int = 42,
):
    """Generate one .rou.xml flow file per sim_day with overnight fill and taper.

    When ``subslots_per_slot`` > 1 and ``spread`` > 0, each full-width CSV slot
    is emitted as several consecutive flows with bounded-uniform sub-slot weights
    (same expected vehicle count as one flat rate; see :func:`_bounded_weights`).
    """
    os.makedirs(out_dir, exist_ok=True)
    slots_sorted = sorted(df["start_time"].unique())
    data_start = time_to_seconds(slots_sorted[0])
    data_end = time_to_seconds(slots_sorted[-1]) + slot_minutes * 60
    vph_factor = 60.0 / slot_minutes
    full_slot_seconds = float(slot_minutes * 60)
    n_days = df["sim_day"].nunique()
    use_variation = int(subslots_per_slot) > 1 and float(spread) > 0.0

    if int_cfg is None:
        int_cfg = {}

    allowed = _allowed_movements(active, int_cfg)

    max_speed_kmh = max(
        (int_cfg.get("approaches", {}).get(d, {}).get("speed_kmh", 50) for d in active),
        default=50,
    )
    max_speed_ms = max_speed_kmh / 3.6

    for day_id in range(n_days):
        day_df = df[df["sim_day"] == day_id].sort_values("start_time")
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set(
            "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd"
        )

        vt = ET.SubElement(root, "vType")
        vt.set("id", "passenger")
        vt.set("accel", "2.6")
        vt.set("decel", "4.5")
        vt.set("sigma", "0.5")
        vt.set("length", "5")
        vt.set("maxSpeed", f"{max_speed_ms:.2f}")

        fid = [0]

        def add_flow(begin, end, from_approach, to_approach, vph):
            if vph < 0.5:
                return
            f = ET.SubElement(root, "flow")
            f.set("id", f"f{fid[0]}")
            f.set("type", "passenger")
            f.set("from", edge_in_id(from_approach))
            f.set("to", edge_out_id(to_approach))
            f.set("begin", str(int(begin)))
            f.set("end", str(int(end)))
            f.set("vehsPerHour", f"{vph:.1f}")
            f.set("departLane", "best")
            f.set("departSpeed", "max")
            fid[0] += 1

        n_allowed = max(len(allowed), 1)

        if data_start > 0:
            for d, movement in sorted(allowed):
                target = TURN_TARGETS[d][movement]
                add_flow(0, data_start, d, target, OVERNIGHT_VPH / n_allowed)

        for row_index, (_, row) in enumerate(day_df.iterrows()):
            begin = time_to_seconds(row["start_time"])
            end = begin + full_slot_seconds
            for d, movement in sorted(allowed):
                target = TURN_TARGETS[d][movement]
                base_vph = row[f"{d}_{movement}"] * vph_factor
                tag = f"{d}_{movement}"
                rng = (
                    _variation_rng(variation_seed, day_id, row_index, tag)
                    if use_variation
                    else None
                )
                _emit_subslot_flows(
                    add_flow,
                    begin,
                    end,
                    d,
                    target,
                    base_vph,
                    full_slot_seconds=full_slot_seconds,
                    subslots_per_slot=subslots_per_slot,
                    spread=spread,
                    rng=rng,
                )

        remaining = SIM_DURATION - data_end
        if remaining > 0:
            last_row = day_df.iloc[-1]
            n_taper = math.ceil(remaining / full_slot_seconds)
            for i in range(n_taper):
                t_frac = i / max(n_taper - 1, 1)
                begin = data_end + i * full_slot_seconds
                end = min(begin + full_slot_seconds, SIM_DURATION)
                taper_row_index = 1_000_000 + i
                for d, movement in sorted(allowed):
                    target = TURN_TARGETS[d][movement]
                    last_vph = last_row[f"{d}_{movement}"] * vph_factor
                    blended = last_vph * (1 - t_frac) + (OVERNIGHT_VPH / n_allowed) * t_frac
                    tag = f"taper{i}_{d}_{movement}"
                    rng = (
                        _variation_rng(variation_seed, day_id, taper_row_index, tag)
                        if use_variation
                        else None
                    )
                    _emit_subslot_flows(
                        add_flow,
                        begin,
                        end,
                        d,
                        target,
                        blended,
                        full_slot_seconds=full_slot_seconds,
                        subslots_per_slot=subslots_per_slot,
                        spread=spread,
                        rng=rng,
                    )

        raw = ET.tostring(root, encoding="unicode")
        lines = minidom.parseString(raw).toprettyxml(indent="  ").split("\n")
        lines = [l for l in lines if not l.startswith("<?xml")]
        path = os.path.join(out_dir, f"flows_day_{day_id:02d}.rou.xml")
        with open(path, "w") as fh:
            fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            fh.write("\n".join(lines))

    if use_variation:
        _mr = 1.0 + 2.0 * float(spread)
        print(
            f"  Demand variation: {int(subslots_per_slot)} sub-slots/slot, spread={float(spread):.3g}, "
            f"max_ratio={_mr:.4g} (seed={variation_seed})"
        )
    print(f"  Written {n_days} SUMO flow files -> {out_dir}")


def write_sumocfg(
    net_file: str, flows_dir: str, out_path: str, begin: int = 27000, end: int = 64800
) -> None:
    """Write a .sumocfg for visual inspection with sumo-gui."""
    route_files = (
        sorted(f for f in os.listdir(flows_dir) if f.endswith(".rou.xml"))
        if os.path.isdir(flows_dir)
        else []
    )
    route_str = ",".join(route_files) if route_files else "flows_day_00.rou.xml"

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        )
        f.write(
            '  xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
        )
        f.write("  <input>\n")
        f.write(f'    <net-file value="{os.path.abspath(net_file)}"/>\n')
        f.write(f'    <route-files value="{route_str}"/>\n')
        f.write("  </input>\n")
        f.write("  <time>\n")
        f.write(f'    <begin value="{begin}"/>\n')
        f.write(f'    <end value="{end}"/>\n')
        f.write("  </time>\n")
        f.write("</configuration>\n")

    print(f"  sumocfg written -> {out_path}")
    print(f"  Run: sumo-gui -c {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess traffic CSV for SUMO + RL")
    parser.add_argument("--csv", required=True, help="Path to traffic count CSV / XLS")
    parser.add_argument("--intersection", required=True, help="Path to intersection.json")
    parser.add_argument("--columns", required=True, help="Path to columns.json")
    parser.add_argument("--out-dir", default="data", help="Root output directory")
    parser.add_argument("--test-days", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--date-mode", choices=("error", "offset", "concat"), default="concat",
    )
    parser.add_argument(
        "--subslots",
        type=int,
        default=5,
        help="Sub-intervals per CSV time slot for SUMO flows (1 = single flat vehsPerHour per slot)",
    )
    parser.add_argument(
        "--demand-spread",
        type=float,
        default=0.85,
        help="Strength of random rate swings between sub-slots; 0 disables variation",
    )
    parser.add_argument(
        "--demand-seed",
        type=int,
        default=None,
        help="RNG seed for demand variation (default: same as --seed)",
    )
    parser.add_argument("--write-sumocfg", action="store_true")
    parser.add_argument("--sumocfg-out", default="simulation.sumocfg")
    args = parser.parse_args()

    print(f"Loading intersection config: {args.intersection}")
    int_cfg = load_intersection(args.intersection)
    active = int_cfg["active_approaches"]
    name = int_cfg["intersection_name"]
    min_red = int_cfg.get("min_red_s", 15)
    print(f"  Intersection : {name}")
    print(f"  Approaches   : {active}")

    print(f"Loading column map: {args.columns}")
    col_map = load_column_map(args.columns)
    slot_minutes = col_map.get("slot_minutes", 15)
    print(f"  Slot duration: {slot_minutes} min")

    print(f"Loading CSV: {args.csv}")
    df = load_csv(args.csv, col_map, active)

    print(f"Assigning simulated day IDs (date-mode={args.date_mode}) …")
    df = assign_sim_days(df, date_mode=args.date_mode)
    n_days = df["sim_day"].nunique()
    n_slots = df["start_time"].nunique()
    print(f"  {n_days} simulated days  |  {n_slots} slots per day")

    if args.test_days >= n_days:
        raise ValueError(
            f"test_days ({args.test_days}) must be < total days ({n_days})"
        )

    processed_dir = os.path.join(args.out_dir, "processed")
    days_dir = os.path.join(processed_dir, "days")
    flows_dir = os.path.join(args.out_dir, "sumo", "flows")

    print("Writing per-day CSVs …")
    write_day_csvs(df, active, days_dir)

    print("Writing SUMO flow files …")
    demand_seed = args.demand_seed if args.demand_seed is not None else args.seed
    write_sumo_flows(
        df,
        active,
        slot_minutes,
        flows_dir,
        int_cfg,
        subslots_per_slot=args.subslots,
        spread=args.demand_spread,
        variation_seed=demand_seed,
    )

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
        "min_red_seconds": min_red,
        "n_days": n_days,
        "n_slots": n_slots,
        "slot_minutes": slot_minutes,
        "day_of_week_map": dow_map,
        "edge_ids": {
            d: {"in": edge_in_id(d), "out": edge_out_id(d)} for d in active
        },
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
        net_dir = os.path.join(args.out_dir, "sumo", "network")
        net_file = os.path.join(net_dir, f"{name}.net.xml")
        if os.path.exists(net_file):
            write_sumocfg(net_file, flows_dir, args.sumocfg_out, begin=27000, end=64800)
        else:
            print(
                f"  Warning: network file not found at {net_file} — skipping sumocfg.\n"
                f"  Run BuildNetwork.py first, then re-run with --write-sumocfg."
            )

    print(f"\nDone. Run the RL trainer with:")
    print(f"  python train.py --data {args.out_dir}/processed/split.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify-bounded-weights":
        verify_bounded_subslot_weights()
        raise SystemExit(0)
    main()
