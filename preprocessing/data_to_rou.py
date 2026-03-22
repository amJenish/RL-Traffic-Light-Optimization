#!/usr/bin/env python3
from __future__ import annotations

"""
Convert intersection count data (Toronto-style format) to SUMO rou.xml.

Expected columns (case-insensitive, underscores optional):
  location_name, start_time, end_time, date
  n_approaching_r, n_approaching_t, n_approaching_l, n_approaching_peds
  s_approaching_r, s_approaching_t, s_approaching_l, s_approaching_peds
  e_approaching_r, e_approaching_t, e_approaching_l, e_approaching_peds
  w_approaching_r, w_approaching_t, w_approaching_l, w_approaching_peds

Output: SUMO routes file with <route> and <flow> elements.
--net and --auto-edges: derive edges from central junction (no manual edge IDs).
Otherwise use --edge-mapping CSV or placeholder edge IDs.
"""

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


# Required columns (normalized: lower, strip)
REQUIRED = {"location_name", "start_time", "end_time", "date"}
APPROACHES = ("n", "s", "e", "w")
MOVEMENTS = ("r", "t", "l")  # right, through, left (vehicle only; peds optional later)


def normalize_col(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.lower().strip())


def _parse_time(s: str) -> float:
    """Parse HH:MM:SS or HH:MM to seconds from midnight."""
    s = str(s).strip()
    parts = s.split(":")
    if len(parts) == 2:
        h, m = int(parts[0]), int(parts[1])
        return h * 3600 + m * 60
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600 + m * 60 + sec
    raise ValueError(f"Invalid time: {s}")


def load_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xls", ".xlsx"):
        try:
            df = pd.read_excel(path)
        except Exception:
            # File may be CSV misnamed as .xls
            df = pd.read_csv(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_excel(path)

    # Normalize column names (handle Unnamed / first column)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    # Drop index/unnamed columns so they don't interfere
    uname = [c for c in df.columns if c.startswith("unnamed")]
    if uname:
        df = df.drop(columns=uname, errors="ignore")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    for approach in APPROACHES:
        for mov in MOVEMENTS:
            col = f"{approach}_approaching_{mov}"
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")


def slug(s: str) -> str:
    """Safe ID fragment from location name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip()).strip("_") or "loc"


def _angle_to_approach(angle_deg: float) -> str:
    """Map angle (degrees, 0=East, 90=North, counterclockwise) to n/s/e/w."""
    # Normalize to [0, 360); East=0, North=90, West=180, South=270
    a = angle_deg % 360
    if a < 45 or a >= 315:
        return "e"
    if 45 <= a < 135:
        return "n"
    if 135 <= a < 225:
        return "w"
    return "s"


_CARDINAL_CENTER = {"e": 0, "n": 90, "w": 180, "s": 270}


def _angular_error(angle_deg: float, card: str) -> float:
    """Smallest angular error from angle_deg to cardinal direction (degrees)."""
    center = _CARDINAL_CENTER[card]
    err = abs((angle_deg % 360) - center)
    return min(err, 360 - err)


def _normalize_turn_dir(d: str) -> str:
    """SUMO dir: r, l, t, s. We use r, t, l; treat s as t (through)."""
    d = (d or "t").lower()
    return "t" if d == "s" else d


def _edge_length_from_elem(e) -> float:
    """Compute edge length from first lane length or shape."""
    lanes = e.findall("lane")
    if lanes and lanes[0].get("length"):
        try:
            return float(lanes[0].get("length"))
        except (TypeError, ValueError):
            pass
    shape = (e.get("shape") or "").strip()
    if shape:
        pts = []
        for p in shape.split():
            try:
                pts.append(tuple(map(float, p.split(","))))
            except ValueError:
                pass
        if len(pts) >= 2:
            return sum(
                math.sqrt((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2)
                for i in range(len(pts) - 1)
            )
    return 0.0


def parse_network(net_path: Path) -> tuple[dict, dict, list, set]:
    """
    Parse SUMO net.xml. Returns (junctions, edges, connections, edge_ids).
    Ignores internal edges (id starting with ':', function="internal") and internal junctions.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()
    junctions = {}
    for j in root.findall(".//junction"):
        jid = j.get("id")
        if not jid or jid.startswith(":"):
            continue
        try:
            inc_lanes_str = j.get("incLanes") or ""
            inc_lanes = len(inc_lanes_str.split()) if inc_lanes_str.strip() else 0
            junctions[jid] = {
                "x": float(j.get("x", 0)),
                "y": float(j.get("y", 0)),
                "type": j.get("type") or "",
                "inc_lanes": inc_lanes,
            }
        except (TypeError, ValueError):
            pass
    edges = {}
    for e in root.findall(".//edge"):
        eid = e.get("id")
        if not eid or eid.startswith(":") or e.get("function") == "internal":
            continue
        if "footway" in (e.get("type") or "").lower() or "footway" in (e.get("id") or "").lower():
            continue
        from_node = e.get("from")
        to_node = e.get("to")
        if from_node and to_node:
            edges[eid] = {
                "from": from_node,
                "to": to_node,
                "shape": e.get("shape"),
                "num_lanes": len(e.findall("lane")),
                "length": _edge_length_from_elem(e),
            }
    connections = []
    for c in root.findall(".//connection"):
        from_e, to_e = c.get("from"), c.get("to")
        if from_e and to_e and not from_e.startswith(":"):
            connections.append((from_e, to_e, _normalize_turn_dir(c.get("dir", "t"))))
    return junctions, edges, connections, set(edges.keys())


# Standard movement mapping: approach -> (R, T, L) -> outgoing direction
# North: R->East, T->South, L->West; South: R->West, T->North, L->East; etc.
_MOVEMENT_TO_OUT: dict[tuple[str, str], str] = {
    ("n", "r"): "e", ("n", "t"): "s", ("n", "l"): "w",
    ("s", "r"): "w", ("s", "t"): "n", ("s", "l"): "e",
    ("e", "r"): "s", ("e", "t"): "w", ("e", "l"): "n",
    ("w", "r"): "n", ("w", "t"): "e", ("w", "l"): "s",
}


def _select_center_junction(
    junctions: dict, edges: dict
) -> tuple[str, int, int]:
    """
    Choose center junction: prefer inc>=4 and out>=4, then traffic_light,
    then max(inc+out), then max(inc_lanes+out_lanes), then max avg edge length.
    Returns (center_id, inc_count, out_count). Raises if no 4-leg junction.
    """
    candidates = []
    for jid in junctions:
        inc_count = sum(1 for e in edges.values() if e["to"] == jid)
        out_count = sum(1 for e in edges.values() if e["from"] == jid)
        if inc_count == 0 or out_count == 0:
            continue
        inc_lanes = junctions[jid].get("inc_lanes", 0) or 0
        out_lanes = sum(
            edges[eid].get("num_lanes", 0) for eid, e in edges.items() if e["from"] == jid
        )
        jtype = (junctions[jid].get("type") or "").strip().lower()
        connected = [e for e in edges.values() if e["from"] == jid or e["to"] == jid]
        avg_len = sum(e.get("length", 0) for e in connected) / len(connected) if connected else 0
        candidates.append(
            (jid, inc_count, out_count, inc_lanes, out_lanes, jtype == "traffic_light", avg_len)
        )
    if not candidates:
        raise ValueError("No junction has both incoming and outgoing real edges. Use --edge-mapping.")
    four_leg = [c for c in candidates if c[1] >= 4 and c[2] >= 4]
    if four_leg:
        pool = four_leg
        traffic_light = [c for c in pool if c[5]]
        if traffic_light:
            pool = traffic_light
        pool.sort(key=lambda c: (-(c[1] + c[2]), -(c[3] + c[4]), -c[6]))
    else:
        candidates.sort(key=lambda c: (-(c[1] + c[2]), -(c[3] + c[4]), -c[6]))
        best = candidates[0]
        raise ValueError(
            "No junction with at least 4 incoming and 4 outgoing edges (network is not a 4-leg intersection). "
            "Best candidate: junction %s with inc_count=%d, out_count=%d. Use --edge-mapping to specify edges manually."
            % (best[0], best[1], best[2])
        )
    best = pool[0]
    return (best[0], best[1], best[2])


def _edge_far_point_and_angle(edge_id: str, edge: dict, junctions: dict, center_id: str, incoming: bool) -> tuple[float, float, float]:
    """Return (x, y, angle_deg) of the far end of the edge from center. For incoming, far = from; for outgoing, far = to."""
    jx, jy = junctions[center_id]["x"], junctions[center_id]["y"]
    other_id = edge["from"] if incoming else edge["to"]
    shape = (edge.get("shape") or "").strip()
    if shape:
        parts = shape.split()
        if parts:
            pt = parts[0] if incoming else parts[-1]
            try:
                x, y = map(float, pt.split(","))
                dx, dy = x - jx, y - jy
                return (x, y, math.degrees(math.atan2(dy, dx)))
            except (ValueError, IndexError):
                pass
    if other_id in junctions:
        x, y = junctions[other_id]["x"], junctions[other_id]["y"]
        dx, dy = x - jx, y - jy
        return (x, y, math.degrees(math.atan2(dy, dx)))
    return (jx, jy, 0.0)


def _pick_four_legs(edge_ids: list[str], edges: dict, junctions: dict, center_id: str, incoming: bool) -> dict[str, str]:
    """Pick exactly 4 edges (one per N/S/E/W): longest candidates, label by geometry; if multiple map to same cardinal, keep smallest angular error."""
    jx, jy = junctions[center_id]["x"], junctions[center_id]["y"]
    candidates: list[tuple[float, float, str]] = []  # dist_sq, angle, eid
    for eid in edge_ids:
        edge = edges.get(eid)
        if not edge:
            continue
        x, y, angle = _edge_far_point_and_angle(eid, edge, junctions, center_id, incoming)
        dist_sq = (x - jx) ** 2 + (y - jy) ** 2
        candidates.append((dist_sq, angle, eid))
    candidates.sort(key=lambda t: -t[0])  # longest first
    by_card: dict[str, list[tuple[float, float, str]]] = {"n": [], "s": [], "e": [], "w": []}
    for dist_sq, angle, eid in candidates:
        card = _angle_to_approach(angle)
        by_card[card].append((_angular_error(angle, card), dist_sq, eid))
    by_approach: dict[str, str] = {}
    for card in APPROACHES:
        if not by_card[card]:
            return {}
        by_card[card].sort(key=lambda t: (t[0], -t[1]))  # smallest error, then longest
        by_approach[card] = by_card[card][0][2]
    return by_approach


def auto_detect_edges(net_path: Path) -> tuple[dict[tuple[str, str, str], list[str]], str, dict]:
    """
    Find center junction (prefer 4-leg, then traffic_light), 4 incoming + 4 outgoing legs by geometry,
    build 12 (approach, movement) -> [in_edge, out_edge]. Returns (mapping, center_id, legs_info).
    Raises ValueError if no 4-leg junction or not exactly 4 legs.
    """
    junctions, edges, _connections, _edge_ids = parse_network(net_path)
    if not junctions or not edges:
        raise ValueError("Network has no non-internal junctions or edges. Use --edge-mapping.")
    center_id, inc_count, out_count = _select_center_junction(junctions, edges)
    in_ids = [eid for eid, e in edges.items() if e["to"] == center_id]
    out_ids = [eid for eid, e in edges.items() if e["from"] == center_id]
    in_by_dir = _pick_four_legs(in_ids, edges, junctions, center_id, incoming=True)
    out_by_dir = _pick_four_legs(out_ids, edges, junctions, center_id, incoming=False)
    if len(in_by_dir) != 4 or len(out_by_dir) != 4:
        raise ValueError(
            "Auto-edges failed: need exactly 4 incoming and 4 outgoing legs at central junction. "
            "Junction %s has inc_count=%d, out_count=%d but geometry yielded %d in, %d out (N/S/E/W). Use --edge-mapping to specify edges manually."
            % (center_id, inc_count, out_count, len(in_by_dir), len(out_by_dir))
        )
    mapping: dict[tuple[str, str, str], list[str]] = {}
    for approach in APPROACHES:
        for movement in MOVEMENTS:
            out_dir = _MOVEMENT_TO_OUT[(approach, movement)]
            if approach not in in_by_dir or out_dir not in out_by_dir:
                raise ValueError("Auto-edges: missing leg for %s %s. Use --edge-mapping." % (approach, movement))
            mapping[(approach, movement)] = [in_by_dir[approach], out_by_dir[out_dir]]
    legs_info = {"in": in_by_dir, "out": out_by_dir, "center": center_id}
    print(
        "Auto-edges: selected junction id=%s  inc_count=%d  out_count=%d  N/S/E/W in=%s  out=%s" % (
            center_id, inc_count, out_count,
            [in_by_dir[a] for a in APPROACHES],
            [out_by_dir[a] for a in APPROACHES],
        ), file=sys.stderr
    )
    return mapping, center_id, legs_info


def _parse_date(d: object) -> object:
    """Return date-like for comparison (keep as-is for pandas)."""
    return d


def build_flows(
    df: pd.DataFrame,
    date_mode: str,
) -> tuple[list[dict], set[tuple[str, str, str]], dict]:
    """
    Aggregate counts by (location, start_time, end_time, date, approach, movement).
    date_mode: error|offset|concat. Returns (flows, route_keys, info={n_days, day_span, first_date}).
    """
    df = df.copy()
    df["_start_sec"] = df["start_time"].apply(_parse_time)
    df["_end_sec"] = df["end_time"].apply(_parse_time)
    df["_location_slug"] = df["location_name"].apply(slug)

    dates = df["date"].dropna().unique()
    try:
        dates_sorted = sorted(dates, key=lambda d: (_parse_date(d),))
    except Exception:
        dates_sorted = list(dates)
    n_days = len(dates_sorted)
    first_date = dates_sorted[0] if n_days else None
    min_start_sec = df["_start_sec"].min()
    max_end_sec = df["_end_sec"].max()
    day_span = max_end_sec - min_start_sec if max_end_sec > min_start_sec else 86400

    if date_mode == "error" and n_days > 1:
        raise ValueError(
            "Multiple distinct dates in input (%d days). Use --date-mode offset or concat, or restrict to single-day data." % n_days
        )

    route_keys = set()
    flows = []
    for _, grp in df.groupby(["location_name", "_location_slug", "start_time", "end_time", "date", "_start_sec", "_end_sec"], dropna=False):
        row = grp.iloc[0]
        loc_slug = row["_location_slug"]
        start_sec = row["_start_sec"]
        end_sec = row["_end_sec"]
        date_val = row["date"]
        try:
            day_ix = dates_sorted.index(date_val) if date_val in list(dates_sorted) else 0
        except (ValueError, TypeError):
            day_ix = 0
        if date_mode == "offset":
            offset = (pd.Timestamp(str(date_val)) - pd.Timestamp(str(first_date))).days * 86400 if first_date is not None else 0
            begin = start_sec + offset
            end = end_sec + offset
        elif date_mode == "concat":
            begin = day_ix * day_span + (start_sec - min_start_sec)
            end = day_ix * day_span + (end_sec - min_start_sec)
        else:
            begin, end = start_sec, end_sec

        for approach in APPROACHES:
            for mov in MOVEMENTS:
                count = grp[f"{approach}_approaching_{mov}"].sum()
                if count <= 0:
                    continue
                route_keys.add((loc_slug, approach, mov))
                flows.append({
                    "route_id": f"r_{loc_slug}_{approach}_{mov}",
                    "flow_id": f"f_{loc_slug}_{approach}_{mov}_{begin:.0f}_{end:.0f}_{len(flows)}",
                    "begin": begin,
                    "end": end,
                    "number": int(round(count)),
                    "loc_slug": loc_slug,
                    "approach": approach,
                    "movement": mov,
                })

    info = {"n_days": n_days, "day_span": day_span, "first_date": first_date}
    return flows, route_keys, info


def load_edge_mapping(path: Path) -> dict[tuple[str, str, str], list[str]]:
    """Load CSV: location_slug, approach, movement, edges (comma or space separated)."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    key_cols = ["location_slug", "approach", "movement"]
    edge_col = "edges"
    for c in ("edge", "edge_ids", "route_edges"):
        if c in df.columns:
            edge_col = c
            break
    if edge_col not in df.columns or not all(k in df.columns for k in key_cols):
        return {}
    mapping = {}
    for _, r in df.iterrows():
        key = (str(r["location_slug"]).strip(), str(r["approach"]).strip().lower(), str(r["movement"]).strip().lower())
        raw = str(r[edge_col]).strip()
        edges = [e.strip() for e in re.split(r"[\s,]+", raw) if e.strip()]
        if len(edges) >= 2:
            mapping[key] = edges
    return mapping


def validate_edge_mapping_complete(route_keys: set[tuple[str, str, str]], edge_mapping: dict[tuple[str, str, str], list[str]]) -> None:
    """Require all (loc_slug, approach, movement) for each location to be in edge_mapping. Raise with summary of missing."""
    loc_slugs = set(k[0] for k in route_keys)
    required = {(loc, a, m) for loc in loc_slugs for a in APPROACHES for m in MOVEMENTS}
    missing = required - set(edge_mapping.keys())
    if not missing:
        return
    by_loc: dict[str, list[tuple[str, str]]] = {}
    for (loc, a, m) in missing:
        by_loc.setdefault(loc, []).append((a, m))
    parts = ["Missing edge-mapping entries (location_slug, approach, movement):"]
    for loc in sorted(by_loc.keys()):
        parts.append("  %s: %s" % (loc, sorted(by_loc[loc])))
    parts.append("Your edge mapping is incomplete. Re-run generate_edge_mapping.py or provide a complete --edge-mapping CSV.")
    raise ValueError("\n".join(parts))


def write_rou_xml(
    out_path: Path,
    flows: list[dict],
    route_keys: set[tuple[str, str, str]],
    edge_mapping: dict[tuple[str, str, str], list[str]],
) -> None:
    def edge_list(loc_slug: str, approach: str, movement: str) -> list[str]:
        key = (loc_slug, approach, movement)
        if key not in edge_mapping:
            raise ValueError(
                "Missing edge mapping for (location_slug=%r, approach=%r, movement=%r). "
                "Your edge mapping is incomplete. Re-run generate_edge_mapping.py or provide a complete --edge-mapping CSV." % (loc_slug, approach, movement)
            )
        return edge_mapping[key]

    with open(out_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('  <vType id="car" vClass="passenger" guiShape="passenger" speedFactor="1" speedDev="0.1"/>\n')

        # Routes (unique by route_id)
        seen_routes = set()
        for fl in flows:
            rid = fl["route_id"]
            if rid in seen_routes:
                continue
            seen_routes.add(rid)
            edges = edge_list(fl["loc_slug"], fl["approach"], fl["movement"])
            f.write(f'  <route id="{rid}" edges="{" ".join(edges)}"/>\n')

        for fl in flows:
            f.write(
                f'  <flow id="{fl["flow_id"]}" type="car" route="{fl["route_id"]}" '
                f'begin="{fl["begin"]}" end="{fl["end"]}" number="{fl["number"]}"/>\n'
            )
        f.write("</routes>\n")


def apply_shift_to_zero(flows: list[dict]) -> tuple[float, float]:
    """Shift so earliest begin is 0; ensure begin>=0 and end>begin (else end=begin+900). Returns (min_begin, max_end) written."""
    if not flows:
        return (0.0, 0.0)
    t0 = min(f["begin"] for f in flows)
    for f in flows:
        f["begin"] -= t0
        f["end"] -= t0
        if f["begin"] < 0:
            f["begin"] = 0.0
        if f["end"] <= f["begin"]:
            f["end"] = f["begin"] + 900.0
    min_begin = min(f["begin"] for f in flows)
    max_end = max(f["end"] for f in flows)
    return (min_begin, max_end)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert intersection count data to SUMO rou.xml")
    parser.add_argument("input", type=Path, help="Input file (.csv, .xls, or .xlsx)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output rou.xml path (default: <input_stem>_rou.xml)")
    parser.add_argument("--net", type=Path, default=None, help="SUMO network file (e.g. network.net.xml)")
    parser.add_argument("--auto-edges", action="store_true", help="Derive route edges from --net (central junction, no manual IDs)")
    parser.add_argument("--edge-mapping", type=Path, default=None, help="CSV: location_slug, approach, movement, edges")
    parser.add_argument("--shift-to-zero", action="store_true", default=True, help="Shift flow begin/end so first flow starts at 0 (default: on)")
    parser.add_argument("--no-shift-to-zero", action="store_false", dest="shift_to_zero", help="Do not shift times")
    parser.add_argument("--date-mode", choices=("error", "offset", "concat"), default="concat",
                        help="Multi-day handling: error=require single date; offset=add (date-first_date).days*86400 to begin/end; concat=stack days by day_span (default: concat)")
    parser.add_argument("--write-sumocfg", action="store_true", help="Write a SUMO config file that references the net and route file")
    parser.add_argument("--sumocfg-out", type=Path, default=Path("simulation.sumocfg"), help="Output path for the SUMO config (default: simulation.sumocfg)")
    parser.add_argument("--net-file-name", type=str, default=None, help="Net filename in sumocfg (default: basename of --net, or network.net.xml)")
    parser.add_argument("--route-file-name", type=str, default=None, help="Route filename in sumocfg (default: basename of -o, or flows.rou.xml)")
    parser.add_argument("--sumo-begin", type=int, default=0, help="Simulation begin time in sumocfg (default: 0)")
    parser.add_argument("--sumo-end", type=int, default=None, help="Simulation end time in sumocfg (default: max flow end + 300)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_rou.xml"

    if args.auto_edges and not args.net:
        print("Error: --auto-edges requires --net.", file=sys.stderr)
        return 1

    try:
        df = load_data(args.input)
        validate_columns(df)

        flows, route_keys, flow_info = build_flows(df, date_mode=args.date_mode)
        n_days, day_span = flow_info["n_days"], flow_info["day_span"]

        edge_mapping: dict[tuple[str, str, str], list[str]] = {}
        auto_legs = None
        if args.auto_edges:
            try:
                by_am, center_id, legs_info = auto_detect_edges(args.net)
                auto_legs = {"center": center_id, "in": legs_info["in"], "out": legs_info["out"]}
                for (loc_slug, approach, mov) in route_keys:
                    edge_mapping[(loc_slug, approach, mov)] = by_am[(approach, mov)]
            except ValueError as e:
                print("Error: %s" % e, file=sys.stderr)
                return 1
        if args.edge_mapping:
            manual = load_edge_mapping(args.edge_mapping)
            for k, v in manual.items():
                edge_mapping[k] = v

        if route_keys and not edge_mapping:
            raise ValueError(
                "Edge mapping required for all routes. Provide --edge-mapping CSV (or --net --auto-edges). "
                "Re-run generate_edge_mapping.py to generate a mapping from your network."
            )
        if route_keys and edge_mapping:
            validate_edge_mapping_complete(route_keys, edge_mapping)
        if args.net and edge_mapping:
            _, _, _, net_edge_ids = parse_network(args.net)
            invalid = []
            for key, edgelist in edge_mapping.items():
                for eid in edgelist:
                    if eid not in net_edge_ids:
                        invalid.append(eid)
            if invalid:
                raise ValueError(
                    "Edges not in net.xml: %s. Check --net file or --edge-mapping CSV." % sorted(set(invalid))
                )

        total_vehicles = sum(f["number"] for f in flows)
        if total_vehicles <= 0:
            raise ValueError(
                "Total vehicles generated is 0. Likely causes: wrong time window, all zero counts, or mapping failure. "
                "Check input counts and --date-mode / --shift-to-zero."
            )

        min_begin, max_end = 0.0, 0.0
        if args.shift_to_zero:
            min_begin, max_end = apply_shift_to_zero(flows)
        else:
            if flows:
                min_begin = min(f["begin"] for f in flows)
                max_end = max(f["end"] for f in flows)
                for f in flows:
                    if f["end"] <= f["begin"]:
                        f["end"] = f["begin"] + 900.0

        write_rou_xml(args.output, flows, route_keys, edge_mapping)
        print(
            "Summary: date_mode=%s  days=%d  flows=%d  vehicles=%d  time_range=[%s, %s]" % (
                args.date_mode, n_days, len(flows), total_vehicles, min_begin, max_end
            ), file=sys.stderr
        )
        if args.write_sumocfg:
            net_name = args.net_file_name if args.net_file_name is not None else (args.net.name if args.net else "network.net.xml")
            route_name = args.route_file_name if args.route_file_name is not None else args.output.name
            sumo_end = args.sumo_end if args.sumo_end is not None else int(max_end) + 300
            sumo_begin = args.sumo_begin
            with open(args.sumocfg_out, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
                f.write('  <input>\n')
                f.write('    <net-file value="%s"/>\n' % net_name)
                f.write('    <route-files value="%s"/>\n' % route_name)
                f.write('  </input>\n')
                f.write('  <time>\n')
                f.write('    <begin value="%d"/>\n' % sumo_begin)
                f.write('    <end value="%d"/>\n' % sumo_end)
                f.write('  </time>\n')
                f.write('</configuration>\n')
            print("Wrote sumocfg: %s" % args.sumocfg_out, file=sys.stderr)
        print("Routes written: %s" % args.output, file=sys.stderr)
        if args.write_sumocfg:
            print("Run: sumo-gui -c %s" % args.sumocfg_out, file=sys.stderr)
        if auto_legs:
            print("Auto-edges: center=%s  in(N/S/E/W)=%s  out(N/S/E/W)=%s" % (
                auto_legs["center"], list(auto_legs["in"].values()), list(auto_legs["out"].values())), file=sys.stderr)
        return 0
    except Exception as e:
        print("Error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
