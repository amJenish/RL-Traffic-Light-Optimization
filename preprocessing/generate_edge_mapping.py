#!/usr/bin/env python3
"""
Generate edge_mapping.csv from a SUMO network so data_to_rou.py can use --edge-mapping
without typing edge IDs by hand.

Reads network.net.xml, finds the central junction (highest degree), assigns
incoming/outgoing edges to N/S/E/W by geometry, and writes a CSV with one row per
(approach, movement) that exists. Works for 3-leg or 4-leg intersections. You may get fewer than 12 rows if the junction
has only 2–3 distinct approach directions (geometry maps multiple legs to the same N/S/E/W).

Usage:
  python3 generate_edge_mapping.py --net network.net.xml -o edge_mapping.csv
  python3 generate_edge_mapping.py --net network.net.xml --location-slug McCowan_Rd_Finch_Ave_E

Then run data_to_rou.py with --edge-mapping edge_mapping.csv (and --net).
"""

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

APPROACHES = ("n", "s", "e", "w")
MOVEMENTS = ("r", "t", "l")

# North: R->East, T->South, L->West; South: R->West, T->North, L->East; etc.
MOVEMENT_TO_OUT = {
    ("n", "r"): "e", ("n", "t"): "s", ("n", "l"): "w",
    ("s", "r"): "w", ("s", "t"): "n", ("s", "l"): "e",
    ("e", "r"): "s", ("e", "t"): "w", ("e", "l"): "n",
    ("w", "r"): "n", ("w", "t"): "e", ("w", "l"): "s",
}


def _angle_to_approach(angle_deg: float) -> str:
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
    center = _CARDINAL_CENTER[card]
    err = abs((angle_deg % 360) - center)
    return min(err, 360 - err)


def _edge_length_from_elem(e) -> float:
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


def _parse_network(net_path: Path):
    """Parse net.xml; return (junctions, edges). Ignores internal and footway."""
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
        fr, to = e.get("from"), e.get("to")
        if fr and to:
            edges[eid] = {
                "from": fr,
                "to": to,
                "shape": e.get("shape"),
                "num_lanes": len(e.findall("lane")),
                "length": _edge_length_from_elem(e),
            }
    return junctions, edges


def _far_point_and_angle(edge: dict, junctions: dict, center_id: str, incoming: bool):
    jx, jy = junctions[center_id]["x"], junctions[center_id]["y"]
    other_id = edge["from"] if incoming else edge["to"]
    shape = (edge.get("shape") or "").strip()
    if shape:
        parts = shape.split()
        if parts:
            pt = parts[0] if incoming else parts[-1]
            try:
                x, y = map(float, pt.split(","))
                return (x, y, math.degrees(math.atan2(y - jy, x - jx)))
            except (ValueError, IndexError):
                pass
    if other_id in junctions:
        x, y = junctions[other_id]["x"], junctions[other_id]["y"]
        return (x, y, math.degrees(math.atan2(y - jy, x - jx)))
    return (jx, jy, 0.0)


def _select_center_junction(junctions: dict, edges: dict) -> tuple[str, int, int]:
    """Prefer inc>=4 and out>=4, then traffic_light, then max(inc+out), then lanes, then avg length. Raises if no 4-leg."""
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
        raise ValueError("No junction has both incoming and outgoing real edges.")
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
            "Best candidate: junction %s with inc_count=%d, out_count=%d." % (best[0], best[1], best[2])
        )
    best = pool[0]
    return (best[0], best[1], best[2])


def _pick_four_legs(edge_ids: list, edges: dict, junctions: dict, center_id: str, incoming: bool) -> dict:
    """Pick exactly 4 edges (one per N/S/E/W); longest first, then smallest angular error per cardinal."""
    jx, jy = junctions[center_id]["x"], junctions[center_id]["y"]
    candidates = []
    for eid in edge_ids:
        edge = edges.get(eid)
        if not edge:
            continue
        x, y, angle = _far_point_and_angle(edge, junctions, center_id, incoming)
        dist_sq = (x - jx) ** 2 + (y - jy) ** 2
        candidates.append((dist_sq, angle, eid))
    candidates.sort(key=lambda t: -t[0])
    by_card = {"n": [], "s": [], "e": [], "w": []}
    for dist_sq, angle, eid in candidates:
        card = _angle_to_approach(angle)
        by_card[card].append((_angular_error(angle, card), dist_sq, eid))
    by_approach = {}
    for card in APPROACHES:
        if not by_card[card]:
            return {}
        by_card[card].sort(key=lambda t: (t[0], -t[1]))
        by_approach[card] = by_card[card][0][2]
    return by_approach


def build_mapping(net_path: Path) -> tuple[list[tuple[str, str, list[str]]], str]:
    """
    Find center junction (prefer 4-leg, traffic_light), pick exactly 4 in/out by geometry, build 12 rows.
    Returns (rows for CSV as (approach, movement, edge_list), center_id). Raises if no 4-leg junction.
    """
    junctions, edges = _parse_network(net_path)
    if not junctions or not edges:
        raise ValueError("Network has no non-internal junctions or edges.")
    center_id, inc_count, out_count = _select_center_junction(junctions, edges)
    in_ids = [eid for eid, e in edges.items() if e["to"] == center_id]
    out_ids = [eid for eid, e in edges.items() if e["from"] == center_id]
    in_by_dir = _pick_four_legs(in_ids, edges, junctions, center_id, incoming=True)
    out_by_dir = _pick_four_legs(out_ids, edges, junctions, center_id, incoming=False)
    if len(in_by_dir) != 4 or len(out_by_dir) != 4:
        raise ValueError(
            "Central junction %s has inc_count=%d, out_count=%d but geometry yielded %d in, %d out (N/S/E/W). "
            "Network may not be a 4-leg intersection." % (center_id, inc_count, out_count, len(in_by_dir), len(out_by_dir))
        )

    rows = []
    for approach in APPROACHES:
        for movement in MOVEMENTS:
            out_dir = MOVEMENT_TO_OUT[(approach, movement)]
            edge_list = [in_by_dir[approach], out_by_dir[out_dir]]
            rows.append((approach, movement, edge_list))
    print(
        "Selected junction id=%s  inc_count=%d  out_count=%d  N/S/E/W in=%s  out=%s" % (
            center_id, inc_count, out_count,
            [in_by_dir[a] for a in APPROACHES],
            [out_by_dir[a] for a in APPROACHES],
        ), file=sys.stderr
    )
    return rows, center_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate edge_mapping.csv from SUMO network.net.xml")
    parser.add_argument("--net", type=Path, required=True, help="SUMO network file (e.g. network.net.xml)")
    parser.add_argument("-o", "--output", type=Path, default=Path("edge_mapping.csv"), help="Output CSV path")
    parser.add_argument("--location-slug", type=str, default="main", help="location_slug to use in CSV (must match your data's location_name slug)")
    args = parser.parse_args()

    if not args.net.exists():
        print("Error: net file not found: %s" % args.net, file=sys.stderr)
        return 1

    try:
        rows, center_id = build_mapping(args.net)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["location_slug", "approach", "movement", "edges"])
            for approach, movement, edge_list in rows:
                w.writerow([args.location_slug, approach, movement, " ".join(edge_list)])
        print("Wrote %d rows to %s (center junction %s)." % (len(rows), args.output, center_id), file=sys.stderr)
        print("Use: python3 data_to_rou.py <data> --net %s --edge-mapping %s" % (args.net, args.output), file=sys.stderr)
        if args.location_slug != "main":
            print("location_slug in CSV is %r; ensure your data's location_name normalizes to this." % args.location_slug, file=sys.stderr)
        return 0
    except Exception as e:
        print("Error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
