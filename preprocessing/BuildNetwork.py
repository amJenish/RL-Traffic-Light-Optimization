"""
build_network.py
----------------
Takes an intersection config (JSON) and generates a SUMO-compatible
network using node, edge, connection, and traffic light files,
then calls netconvert to produce the final .net.xml.

Merged from SUMO-Demand-Generation-Pipeline/generate_edge_mapping.py:
  - write_edge_mapping() — generates edge_mapping.csv after netconvert so
    data_to_rou.py can be used with the built network without typing edge IDs.

Usage:
    python build_network.py --config intersection_config.json
    python build_network.py --config intersection_config.json --out-dir data/sumo/network

Config format (produced by the intersection configurator widget):
{
  "intersection_name": "McCowan_Finch",
  "approaches": {
    "N": { "lanes": { "through": 2, "right": 1, "left": 1 }, "speed_kmh": 50 },
    "S": { "lanes": { "through": 2, "right": 1, "left": 1 }, "speed_kmh": 50 },
    "E": { "lanes": { "through": 2, "right": 1, "left": 1 }, "speed_kmh": 50 },
    "W": { "lanes": { "through": 2, "right": 1, "left": 1 }, "speed_kmh": 50 }
  },
  "active_approaches": ["N", "S", "E", "W"],
  "phases": "4",
  "min_red_s": 15,
  "amber_s": 3,
  "cycle_s": 120,
  "edge_length_m": 200
}

Phases:
  "2" → NS-all / EW-all
  "4" → NS-through+right / NS-left / EW-through+right / EW-left
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom


# GEOMETRY — junction positions

# Centre junction at origin. Approach nodes placed at ±edge_length in each
# cardinal direction.

DIRECTION_VECTORS = {
    "N": ( 0,  1),
    "S": ( 0, -1),
    "E": ( 1,  0),
    "W": (-1,  0),
}

# Opposite approach for each direction (where vehicles exit)
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# Turn targets: from approach X, right/left are from the driver's perspective.
#   N approach → traveling South:  right=W, left=E
#   S approach → traveling North:  right=E, left=W
#   E approach → traveling West:   right=N, left=S
#   W approach → traveling East:   right=S, left=N
TURN_TARGETS = {
    "N": {"right": "W", "through": "S", "left": "E"},
    "S": {"right": "E", "through": "N", "left": "W"},
    "E": {"right": "N", "through": "W", "left": "S"},
    "W": {"right": "S", "through": "E", "left": "N"},
}



# PHASE DEFINITIONS


def _has_movement(approaches: dict, active: list[str], group: list[str],
                  movement: str) -> bool:
    """True if any approach in *group* that is also *active* has lanes for *movement*.
    When no lane config is provided for an approach, the movement is assumed to exist."""
    for d in group:
        if d not in active:
            continue
        app = approaches.get(d)
        if app is None:
            return True
        if app.get("lanes", {}).get(movement, 0) > 0:
            return True
    return False


def build_phases(active: list[str], mode: str, amber_s: int, min_green_s: int,
                 cycle_s: int, approaches: dict | None = None) -> list[dict]:
    """
    Build phase specs driven by the intersection config.

    Only includes a dedicated left-turn phase (in 4-phase mode) if at least
    one approach in the group actually has left-turn lanes.  Only includes
    a group (NS / EW) if at least one of its approaches is active.
    """
    if approaches is None:
        approaches = {}

    ns_dirs = [d for d in ("N", "S") if d in active]
    ew_dirs = [d for d in ("E", "W") if d in active]

    n_green_phases = 0
    if ns_dirs:
        n_green_phases += 1
    if ew_dirs:
        n_green_phases += 1
    if mode == "4":
        if ns_dirs and _has_movement(approaches, active, ns_dirs, "left"):
            n_green_phases += 1
        if ew_dirs and _has_movement(approaches, active, ew_dirs, "left"):
            n_green_phases += 1

    n_green_phases = max(n_green_phases, 1)
    green_time = (cycle_s - n_green_phases * amber_s) // n_green_phases
    green_time = max(green_time, min_green_s)
    green_time = max(green_time, 10)

    phases = []

    if mode == "2":
        if ns_dirs:
            grp = []
            for d in ns_dirs:
                for m in ("through", "right", "left"):
                    grp.append((d, m))
            phases.append({"name": "NS_green", "green_groups": grp,
                           "duration": green_time})
            phases.append({"name": "NS_amber", "green_groups": [],
                           "amber_groups": [(d, "all") for d in ns_dirs],
                           "duration": amber_s})

        if ew_dirs:
            grp = []
            for d in ew_dirs:
                for m in ("through", "right", "left"):
                    grp.append((d, m))
            phases.append({"name": "EW_green", "green_groups": grp,
                           "duration": green_time})
            phases.append({"name": "EW_amber", "green_groups": [],
                           "amber_groups": [(d, "all") for d in ew_dirs],
                           "duration": amber_s})

    else:  # 4-phase
        if ns_dirs:
            thru_grp = []
            for d in ns_dirs:
                thru_grp.extend([(d, "through"), (d, "right")])
            phases.append({"name": "NS_thru", "green_groups": thru_grp,
                           "duration": green_time})
            phases.append({"name": "NS_amber1", "green_groups": [],
                           "amber_groups": [(d, "all") for d in ns_dirs],
                           "duration": amber_s})

            if _has_movement(approaches, active, ns_dirs, "left"):
                left_grp = [(d, "left") for d in ns_dirs]
                phases.append({"name": "NS_left", "green_groups": left_grp,
                               "duration": green_time})
                phases.append({"name": "NS_amber2", "green_groups": [],
                               "amber_groups": [(d, "left") for d in ns_dirs],
                               "duration": amber_s})

        if ew_dirs:
            thru_grp = []
            for d in ew_dirs:
                thru_grp.extend([(d, "through"), (d, "right")])
            phases.append({"name": "EW_thru", "green_groups": thru_grp,
                           "duration": green_time})
            phases.append({"name": "EW_amber1", "green_groups": [],
                           "amber_groups": [(d, "all") for d in ew_dirs],
                           "duration": amber_s})

            if _has_movement(approaches, active, ew_dirs, "left"):
                left_grp = [(d, "left") for d in ew_dirs]
                phases.append({"name": "EW_left", "green_groups": left_grp,
                               "duration": green_time})
                phases.append({"name": "EW_amber2", "green_groups": [],
                               "amber_groups": [(d, "left") for d in ew_dirs],
                               "duration": amber_s})

    return phases

def write_nodes(cfg: dict, out_dir: str) -> str:
    active   = cfg["active_approaches"]
    length   = cfg.get("edge_length_m", 200)
    name     = cfg["intersection_name"]

    root = ET.Element("nodes")

    # Centre junction
    junc = ET.SubElement(root, "node")
    junc.set("id",   "J_centre")
    junc.set("x",    "0")
    junc.set("y",    "0")
    junc.set("type", "traffic_light")

    # Approach/exit nodes
    for d in active:
        dx, dy = DIRECTION_VECTORS[d]
        # Approach node (where vehicles enter from)
        n = ET.SubElement(root, "node")
        n.set("id",   f"node_{d}_in")
        n.set("x",    str(dx * length))
        n.set("y",    str(dy * length))
        n.set("type", "priority")

    path = os.path.join(out_dir, f"{name}.nod.xml")
    _write_xml(root, path)
    return path



# EDGE FILE (.edg.xml)

def _total_lanes(approach_cfg: dict) -> int:
    lanes = approach_cfg["lanes"]
    return lanes.get("through", 1) + lanes.get("right", 0) + lanes.get("left", 0)


def write_edges(cfg: dict, out_dir: str) -> str:
    active   = cfg["active_approaches"]
    name     = cfg["intersection_name"]
    length   = cfg.get("edge_length_m", 200)

    root = ET.Element("edges")

    for d in active:
        app  = cfg["approaches"][d]
        spd  = app["speed_kmh"] / 3.6  # convert to m/s
        n_in = _total_lanes(app)

        # Inbound edge: approach node → centre junction
        e_in = ET.SubElement(root, "edge")
        e_in.set("id",     f"edge_{d}_in")
        e_in.set("from",   f"node_{d}_in")
        e_in.set("to",     "J_centre")
        e_in.set("numLanes", str(n_in))
        e_in.set("speed",  f"{spd:.2f}")
        e_in.set("length", str(length))

        # Outbound edge: centre junction → approach node (exit)
        # Use 2 lanes for outbound by default (through + right merge)
        e_out = ET.SubElement(root, "edge")
        e_out.set("id",     f"edge_{d}_out")
        e_out.set("from",   "J_centre")
        e_out.set("to",     f"node_{d}_in")
        e_out.set("numLanes", str(max(2, n_in - 1)))
        e_out.set("speed",  f"{spd:.2f}")
        e_out.set("length", str(length))

    path = os.path.join(out_dir, f"{name}.edg.xml")
    _write_xml(root, path)
    return path



# CONNECTION FILE (.con.xml)

def write_connections(cfg: dict, out_dir: str) -> str:
    active = cfg["active_approaches"]
    name   = cfg["intersection_name"]
    root   = ET.Element("connections")

    for d in active:
        lanes_cfg = cfg["approaches"][d]["lanes"]
        lane_idx  = 0

        # Assign lanes: right → through → left (right-most = lane 0 in SUMO)
        lane_map = {}
        for movement in ["right", "through", "left"]:
            count = lanes_cfg.get(movement, 0)
            for i in range(count):
                lane_map[f"{movement}_{i}"] = lane_idx
                lane_idx += 1

        targets = TURN_TARGETS[d]
        out_lane = 0

        for movement in ["right", "through", "left"]:
            count = lanes_cfg.get(movement, 0)
            if count == 0:
                continue
            target_dir = targets[movement]
            if target_dir not in active:
                continue  # exit direction not present (T-intersection)
            for i in range(count):
                src_lane = lane_map[f"{movement}_{i}"]
                con = ET.SubElement(root, "connection")
                con.set("from",     f"edge_{d}_in")
                con.set("to",       f"edge_{target_dir}_out")
                con.set("fromLane", str(src_lane))
                con.set("toLane",   str(min(out_lane, max(0,
                    _total_lanes(cfg["approaches"].get(target_dir, {"lanes":{"through":1}})) - 2))))

    path = os.path.join(out_dir, f"{name}.con.xml")
    _write_xml(root, path)
    return path



# TRAFFIC LIGHT FILE (.tll.xml)

def write_tll(cfg: dict, out_dir: str) -> str:
    active   = cfg["active_approaches"]
    name     = cfg["intersection_name"]
    mode       = str(cfg.get("phases", "2"))
    amber_s    = cfg.get("amber_s", 3)
    min_green  = cfg.get("min_green_s", 15)
    cycle_s    = cfg.get("cycle_s", 120)

    phases = build_phases(active, mode, amber_s, min_green, cycle_s,
                          approaches=cfg.get("approaches", {}))

    # Build ordered link list: (from_edge, to_edge, from_lane)
    # This defines the state string order
    links = []
    for d in active:
        lanes_cfg = cfg["approaches"][d]["lanes"]
        lane_idx  = 0
        for movement in ["right", "through", "left"]:
            count = lanes_cfg.get(movement, 0)
            target_dir = TURN_TARGETS[d][movement]
            if target_dir not in active:
                lane_idx += count
                continue
            for i in range(count):
                links.append({
                    "from":      f"edge_{d}_in",
                    "to":        f"edge_{target_dir}_out",
                    "from_lane": lane_idx,
                    "approach":  d,
                    "movement":  movement,
                })
                lane_idx += 1

    n_links = len(links)

    def state_for_phase(phase: dict) -> str:
        green_set = set()
        amber_set = set()
        for grp in phase.get("green_groups", []):
            app, mov = grp
            if mov == "all":
                for m in ["through", "right", "left"]:
                    green_set.add((app, m))
            else:
                green_set.add((app, mov))
        for grp in phase.get("amber_groups", []):
            app, mov = grp
            if mov in ("all", "thru"):
                for m in ["through", "right", "left"]:
                    amber_set.add((app, m))
            else:
                amber_set.add((app, mov))

        state = []
        for lnk in links:
            key = (lnk["approach"], lnk["movement"])
            if key in green_set:
                # Through movements get G (priority), turns get g (yield)
                if lnk["movement"] == "through":
                    state.append("G")
                else:
                    state.append("g")
            elif key in amber_set:
                state.append("y")
            else:
                state.append("r")
        return "".join(state)

    root = ET.Element("tlLogics")
    tl   = ET.SubElement(root, "tlLogic")
    tl.set("id",       "J_centre")
    tl.set("type",     "static")
    tl.set("programID","0")
    tl.set("offset",   "0")

    for ph in phases:
        state = state_for_phase(ph)
        p = ET.SubElement(tl, "phase")
        p.set("duration", str(ph["duration"]))
        p.set("state",    state)
        p.set("name",     ph["name"])

    path = os.path.join(out_dir, f"{name}.tll.xml")
    _write_xml(root, path)
    return path



# EDGE MAPPING  (from generate_edge_mapping.py)

def write_edge_mapping(cfg: dict, out_dir: str) -> str:
    """
    Generate edge_mapping.csv alongside the network files.

    Computes all 12 (approach × movement) → [in_edge, out_edge] pairs
    directly from the intersection config using the known naming convention
    edge_{D}_in / edge_{D}_out produced by write_edges().

    This file can then be passed to preprocessing/data_to_rou.py via
    --edge-mapping so that tool can build rou.xml files without needing
    to re-inspect the network geometry.

    Ported from SUMO-Demand-Generation-Pipeline/generate_edge_mapping.py.
    """
    active  = cfg["active_approaches"]
    name    = cfg["intersection_name"]
    slug    = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    mov_abbr = {"through": "t", "right": "r", "left": "l"}

    rows = []
    for d in active:
        for movement, out_direction in TURN_TARGETS[d].items():
            if out_direction not in active:
                continue
            rows.append({
                "location_slug": slug,
                "approach":      d.lower(),
                "movement":      mov_abbr[movement],
                "edges":         f"edge_{d}_in edge_{out_direction}_out",
            })

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "edge_mapping.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["location_slug", "approach", "movement", "edges"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Edge mapping written -> {path}")
    print(f"  Use with: python preprocessing/data_to_rou.py <csv> "
          f"--net {out_dir}/{name}.net.xml --edge-mapping {path}")
    return path


# NETCONVERT CALL

def run_netconvert(cfg: dict, out_dir: str,
                   nod: str, edg: str, con: str, tll: str) -> str:
    """
    Step 1: Run netconvert WITHOUT our TLL file so it auto-generates
            correct state strings matching its own connection ordering.
    Step 2: Post-process the output to apply our timing constraints.
    """
    name    = cfg["intersection_name"]
    out_net = os.path.join(out_dir, f"{name}.net.xml")

    min_green = cfg.get("min_green_s", 15)
    max_green = cfg.get("max_green_s", 90)
    amber_s   = cfg.get("amber_s", 3)
    mode      = str(cfg.get("phases", "2"))

    # Step 1 — let netconvert auto-generate TLL with correct state strings
    cmd = [
        "netconvert",
        "--node-files",       nod,
        "--edge-files",       edg,
        "--connection-files", con,
        "--output-file",      out_net,
        "--no-turnarounds",   "true",
        "--tls.guess",        "true",    # auto-generate TLS
        "--tls.cycle.time",   str(cfg.get("cycle_s", 120)),
        "--junctions.join",   "false",
        "--verbose",          "false",
    ]

    print(f"  Running netconvert (auto TLS generation)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("netconvert stderr:", result.stderr[-800:])
        raise RuntimeError("netconvert failed — see above")

    # Step 2 — post-process: apply our timing constraints to phases
    _apply_timing_constraints(out_net, min_green, max_green, amber_s, mode)

    # Step 3 — generate edge_mapping.csv for use with data_to_rou.py
    write_edge_mapping(cfg, out_dir)

    print(f"  Network written -> {out_net}")
    return out_net


def _apply_timing_constraints(net_path: str, min_green: int,
                               max_green: int, amber_s: int, mode: str):
    """
    Read the auto-generated .net.xml, adjust phase durations to respect
    min_green and max_green constraints, write back.
    """
    tree = ET.parse(net_path)
    root = tree.getroot()

    for tl in root.findall(".//tlLogic"):
        phases = tl.findall("phase")
        if not phases:
            continue

        for phase in phases:
            state    = phase.get("state", "")
            duration = int(phase.get("duration", 30))

            # Identify phase type by dominant signal character
            greens  = state.count("G") + state.count("g")
            yellows = state.count("y") + state.count("Y")
            total   = len(state)

            if yellows > greens:
                # Amber phase — enforce amber_s
                phase.set("duration", str(amber_s))
            elif greens > 0:
                # Green phase — clamp between min and max
                new_dur = max(min_green, min(max_green, duration))
                phase.set("duration", str(new_dur))

        print(f"  TLS '{tl.get('id')}': {len(phases)} phases adjusted")
        for p in phases:
            print(f"    duration={p.get('duration'):>4s}  state={p.get('state')}")

    tree.write(net_path, encoding="unicode", xml_declaration=True)


# XML HELPER

def _write_xml(root: ET.Element, path: str):
    raw    = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines  = [l for l in pretty.split("\n") if not l.startswith("<?xml")]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write("\n".join(lines))
    print(f"  Written -> {path}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Build SUMO intersection network from config")
    parser.add_argument("--config",  required=True, help="Path to intersection_config.json")
    parser.add_argument("--out-dir", default="data/sumo/network",
                        help="Output directory (default: data/sumo/network)")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    name = cfg["intersection_name"]
    print(f"\nBuilding network: {name}")
    print(f"  Approaches : {cfg['active_approaches']}")
    print(f"  Phases     : {cfg.get('phases','2')}-phase")
    print(f"  Min green  : {cfg.get('min_green_s',15)}s")
    print(f"  Max green  : {cfg.get('max_green_s',90)}s")
    print(f"  Amber      : {cfg.get('amber_s',3)}s")
    print(f"  Cycle      : {cfg.get('cycle_s',120)}s")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    nod = write_nodes(cfg, args.out_dir)
    edg = write_edges(cfg, args.out_dir)
    con = write_connections(cfg, args.out_dir)
    tll = write_tll(cfg, args.out_dir)  # kept for reference/backup only

    try:
        net = run_netconvert(cfg, args.out_dir, nod, edg, con, tll)
        print(f"\nDone. Load in SUMO with: sumo-gui -n {net}")
    except FileNotFoundError:
        print("\nnetconvert not found — intermediate files written successfully.")
        print("Run this command on machine once SUMO is installed:")
        print(f"  netconvert --node-files {nod} --edge-files {edg} "
              f"--connection-files {con} "
              f"--output-file {args.out_dir}/{name}.net.xml "
              f"--no-turnarounds true --tls.guess true "
              f"--tls.cycle.time {cfg.get('cycle_s', 120)}")


if __name__ == "__main__":
    main()