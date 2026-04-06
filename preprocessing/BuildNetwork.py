"""
BuildNetwork.py — generates SUMO node, edge, connection, and traffic-light files
from intersection.json, then calls netconvert to produce the final .net.xml.
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


DIRECTION_VECTORS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# SUMO assigns one signal character per controlled link in junction ``incLanes`` order:
# N → E → S → W (see ``J_centre`` in the built ``.net.xml``). TLS ``state`` strings must match.
SUMO_TLS_LINK_APPROACH_ORDER = ("N", "E", "S", "W")

TURN_TARGETS = {
    "N": {"right": "W", "through": "S", "left": "E"},
    "S": {"right": "E", "through": "N", "left": "W"},
    "E": {"right": "N", "through": "W", "left": "S"},
    "W": {"right": "S", "through": "E", "left": "N"},
}


def _has_movement(
    approaches: dict, active: list[str], group: list[str], movement: str
) -> bool:
    """True if any active approach in the group has lanes for this movement."""
    for d in group:
        if d not in active:
            continue
        app = approaches.get(d)
        if app is None:
            return True
        if app.get("lanes", {}).get(movement, 0) > 0:
            return True
    return False


# Nominal SUMO duration for green service rows; dwell is TraCI-controlled (setPhase).
STATIC_GREEN_PHASE_DURATION_S = 999_999

_VALID_TLS_CHARS = frozenset("GgyruoO")


def _sanitize_tls_state_string(state: str) -> str:
    """SUMO TLS state characters only; unknown → 'r' (Section 3.G)."""
    out: list[str] = []
    for ch in state:
        out.append(ch if ch in _VALID_TLS_CHARS else "r")
    return "".join(out)


def _movement_pairs_for_approach(
    approaches: dict, active: list[str], d: str
) -> list[tuple[str, str]]:
    """(approach, movement) pairs that exist for one leg (link order: R,T,L)."""
    out: list[tuple[str, str]] = []
    if d not in active:
        return out
    app = approaches.get(d)
    for m in ("right", "through", "left"):
        if app is None:
            out.append((d, m))
        elif app.get("lanes", {}).get(m, 0) > 0:
            out.append((d, m))
    return out


def build_phases(
    active: list[str],
    mode: str,
    amber_s: int,
    min_green_s: int,
    cycle_s: int,
    approaches: dict | None = None,
    left_turn_mode: str = "permissive",
    protected_approaches: list[str] | None = None,
) -> list[dict]:
    """
    Green **service** phases only (no yellow rows in the TLS program).

    Yellow / all-red clearance between greens is applied in TraCI (Agent) using
    ``setRedYellowGreenState`` for ``yellow_duration_s`` from config.json.

    ``left_turn_mode``:
      - ``permissive`` (default): lefts with through/right on the same street
        (permissive / Canadian-style: may enter on green and wait to turn).
      - ``protected``: lead-lead split — protected left bar, then through+right,
        per street, when left lanes exist.
      - ``protected_some``: like protected, but only for directions listed in
        ``protected_approaches`` (with left ≥ 1); others keep permissive lefts.

    ``phases`` \"2\"`` → NS_service + EW_service.
    ``phases`` \"4`` + permissive → same two bars (documented in console).
    ``phases`` \"4`` + protected → NS_left?, NS_through, EW_left?, EW_through.
    """
    _ = amber_s, min_green_s, cycle_s  # documented in intersection.json
    if approaches is None:
        approaches = {}

    ns_dirs = [d for d in ("N", "S") if d in active]
    ew_dirs = [d for d in ("E", "W") if d in active]
    lt_raw = (left_turn_mode or "permissive").strip().lower()
    if lt_raw not in ("permissive", "protected", "protected_some"):
        lt_raw = "permissive"

    prot_set = set(protected_approaches or [])
    if lt_raw != "protected_some":
        prot_set = set()

    use_permissive_lefts = mode == "2" or lt_raw == "permissive"
    if mode == "4" and lt_raw == "permissive":
        print(
            "  Note: phases=4 with left_turn_mode=permissive uses 2 green bars "
            "(NS_service, EW_service); lefts are permissive with through/right."
        )

    dur = STATIC_GREEN_PHASE_DURATION_S
    phases: list[dict] = []

    def _append_green(name: str, pairs: list[tuple[str, str]]) -> None:
        if pairs:
            phases.append({"name": name, "green_groups": pairs, "duration": dur})

    def _dirs_protected_left(street_dirs: list[str]) -> list[str]:
        out: list[str] = []
        for d in street_dirs:
            if d not in prot_set:
                continue
            if approaches.get(d) is None:
                out.append(d)
            elif approaches.get(d, {}).get("lanes", {}).get("left", 0) > 0:
                out.append(d)
        return out

    def _street_through_pairs(
        street_dirs: list[str], skip_left_for: set[str]
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for d in street_dirs:
            for m in ("right", "through"):
                if approaches.get(d) is None:
                    pairs.append((d, m))
                elif approaches.get(d, {}).get("lanes", {}).get(m, 0) > 0:
                    pairs.append((d, m))
            if d in skip_left_for:
                continue
            if approaches.get(d) is None:
                pairs.append((d, "left"))
            elif approaches.get(d, {}).get("lanes", {}).get("left", 0) > 0:
                pairs.append((d, "left"))
        return pairs

    if lt_raw == "protected_some":
        if ns_dirs:
            pl_ns = _dirs_protected_left(ns_dirs)
            if pl_ns:
                _append_green("NS_left_prot", [(d, "left") for d in pl_ns])
            skip = set(pl_ns)
            thru_ns = _street_through_pairs(ns_dirs, skip)
            _append_green("NS_service", thru_ns)
        if ew_dirs:
            pl_ew = _dirs_protected_left(ew_dirs)
            if pl_ew:
                _append_green("EW_left_prot", [(d, "left") for d in pl_ew])
            skip_e = set(pl_ew)
            thru_ew = _street_through_pairs(ew_dirs, skip_e)
            _append_green("EW_service", thru_ew)
    elif use_permissive_lefts:
        if ns_dirs:
            grp_ns: list[tuple[str, str]] = []
            for d in ns_dirs:
                grp_ns.extend(_movement_pairs_for_approach(approaches, active, d))
            _append_green("NS_service", grp_ns)
        if ew_dirs:
            grp_ew: list[tuple[str, str]] = []
            for d in ew_dirs:
                grp_ew.extend(_movement_pairs_for_approach(approaches, active, d))
            _append_green("EW_service", grp_ew)
    else:
        if ns_dirs:
            if _has_movement(approaches, active, ns_dirs, "left"):
                _append_green(
                    "NS_left",
                    [
                        (d, "left")
                        for d in ns_dirs
                        if approaches.get(d) is None
                        or approaches.get(d, {}).get("lanes", {}).get("left", 0) > 0
                    ],
                )
            thru_ns: list[tuple[str, str]] = []
            for d in ns_dirs:
                for m in ("right", "through"):
                    if approaches.get(d) is None:
                        thru_ns.append((d, m))
                    elif approaches.get(d, {}).get("lanes", {}).get(m, 0) > 0:
                        thru_ns.append((d, m))
            _append_green("NS_through", thru_ns)

        if ew_dirs:
            if _has_movement(approaches, active, ew_dirs, "left"):
                _append_green(
                    "EW_left",
                    [
                        (d, "left")
                        for d in ew_dirs
                        if approaches.get(d) is None
                        or approaches.get(d, {}).get("lanes", {}).get("left", 0) > 0
                    ],
                )
            thru_ew: list[tuple[str, str]] = []
            for d in ew_dirs:
                for m in ("right", "through"):
                    if approaches.get(d) is None:
                        thru_ew.append((d, m))
                    elif approaches.get(d, {}).get("lanes", {}).get(m, 0) > 0:
                        thru_ew.append((d, m))
            _append_green("EW_through", thru_ew)

    phases = [p for p in phases if p.get("green_groups")]
    if not phases:
        raise ValueError("build_phases produced no green phases — check active_approaches")
    return phases


def inject_tll_into_net(net_path: str, tll_path: str, tls_id: str = "J_centre") -> None:
    """Replace ``tlLogic`` in ``net.xml`` with the program from ``*.tll.xml``."""
    tll_tree = ET.parse(tll_path)
    tll_root = tll_tree.getroot()
    source_tl = None
    for tl in tll_root.findall("tlLogic"):
        if tl.get("id") == tls_id:
            source_tl = tl
            break
    if source_tl is None:
        all_tl = tll_root.findall("tlLogic")
        if not all_tl:
            raise ValueError(f"No tlLogic in {tll_path}")
        source_tl = all_tl[0]

    new_tl = ET.fromstring(ET.tostring(source_tl, encoding="unicode"))

    net_tree = ET.parse(net_path)
    net_root = net_tree.getroot()
    for child in list(net_root):
        if child.tag == "tlLogic" and child.get("id") == tls_id:
            net_root.remove(child)

    # SUMO parses net.xml in order: <connection tl="…"> requires tlLogic to appear first.
    insert_idx = None
    for i, child in enumerate(net_root):
        if child.tag == "connection":
            insert_idx = i
            break
    if insert_idx is not None:
        net_root.insert(insert_idx, new_tl)
    else:
        net_root.append(new_tl)

    net_tree.write(net_path, encoding="unicode", xml_declaration=True)
    print(f"  Injected tlLogic '{tls_id}' from {os.path.basename(tll_path)} into net.xml")


def write_nodes(cfg: dict, out_dir: str) -> str:
    """Generate .nod.xml with centre junction and one node per active approach."""
    active = cfg["active_approaches"]
    length = cfg.get("edge_length_m", 200)
    name = cfg["intersection_name"]

    root = ET.Element("nodes")

    junc = ET.SubElement(root, "node")
    junc.set("id", "J_centre")
    junc.set("x", "0")
    junc.set("y", "0")
    junc.set("type", "traffic_light")

    for d in active:
        dx, dy = DIRECTION_VECTORS[d]
        n = ET.SubElement(root, "node")
        n.set("id", f"node_{d}_in")
        n.set("x", str(dx * length))
        n.set("y", str(dy * length))
        n.set("type", "priority")

    path = os.path.join(out_dir, f"{name}.nod.xml")
    _write_xml(root, path)
    return path


def _total_lanes(approach_cfg: dict) -> int:
    lanes = approach_cfg["lanes"]
    return lanes.get("through", 1) + lanes.get("right", 0) + lanes.get("left", 0)


def write_edges(cfg: dict, out_dir: str) -> str:
    """Generate .edg.xml with inbound and outbound edges per approach."""
    active = cfg["active_approaches"]
    name = cfg["intersection_name"]
    length = cfg.get("edge_length_m", 200)

    root = ET.Element("edges")

    for d in active:
        app = cfg["approaches"][d]
        spd = app["speed_kmh"] / 3.6
        n_in = _total_lanes(app)

        e_in = ET.SubElement(root, "edge")
        e_in.set("id", f"edge_{d}_in")
        e_in.set("from", f"node_{d}_in")
        e_in.set("to", "J_centre")
        e_in.set("numLanes", str(n_in))
        e_in.set("speed", f"{spd:.2f}")
        e_in.set("length", str(length))

        e_out = ET.SubElement(root, "edge")
        e_out.set("id", f"edge_{d}_out")
        e_out.set("from", "J_centre")
        e_out.set("to", f"node_{d}_in")
        e_out.set("numLanes", str(max(2, n_in - 1)))
        e_out.set("speed", f"{spd:.2f}")
        e_out.set("length", str(length))

    path = os.path.join(out_dir, f"{name}.edg.xml")
    _write_xml(root, path)
    return path


def write_connections(cfg: dict, out_dir: str) -> str:
    """Generate .con.xml mapping inbound lanes to outbound edges via turn targets."""
    active = cfg["active_approaches"]
    name = cfg["intersection_name"]
    root = ET.Element("connections")

    for d in active:
        lanes_cfg = cfg["approaches"][d]["lanes"]
        lane_idx = 0

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
                continue
            for i in range(count):
                src_lane = lane_map[f"{movement}_{i}"]
                con = ET.SubElement(root, "connection")
                con.set("from", f"edge_{d}_in")
                con.set("to", f"edge_{target_dir}_out")
                con.set("fromLane", str(src_lane))
                con.set(
                    "toLane",
                    str(
                        min(
                            out_lane,
                            max(
                                0,
                                _total_lanes(
                                    cfg["approaches"].get(
                                        target_dir, {"lanes": {"through": 1}}
                                    )
                                )
                                - 2,
                            ),
                        )
                    ),
                )

    path = os.path.join(out_dir, f"{name}.con.xml")
    _write_xml(root, path)
    return path


def write_tll(cfg: dict, out_dir: str) -> str:
    """Generate .tll.xml traffic light logic with state strings matching link order."""
    active = cfg["active_approaches"]
    name = cfg["intersection_name"]
    mode = str(cfg.get("phases", "2"))
    amber_s = cfg.get("amber_s", 3)
    min_green = cfg.get("min_green_s", 15)
    cycle_s = cfg.get("cycle_s", 120)
    lt_mode = cfg.get("left_turn_mode", "permissive")
    prot = cfg.get("protected_approaches")
    if not isinstance(prot, list):
        prot = None

    phases = build_phases(
        active,
        mode,
        amber_s,
        min_green,
        cycle_s,
        approaches=cfg.get("approaches", {}),
        left_turn_mode=str(lt_mode),
        protected_approaches=prot,
    )

    links = []
    for d in SUMO_TLS_LINK_APPROACH_ORDER:
        if d not in active:
            continue
        lanes_cfg = cfg["approaches"][d]["lanes"]
        lane_idx = 0
        for movement in ["right", "through", "left"]:
            count = lanes_cfg.get(movement, 0)
            target_dir = TURN_TARGETS[d][movement]
            if target_dir not in active:
                lane_idx += count
                continue
            for i in range(count):
                links.append({
                    "from": f"edge_{d}_in",
                    "to": f"edge_{target_dir}_out",
                    "from_lane": lane_idx,
                    "approach": d,
                    "movement": movement,
                })
                lane_idx += 1

    n_links = len(links)

    def state_for_phase(phase: dict) -> str:
        green_set = set()
        for grp in phase.get("green_groups", []):
            app, mov = grp
            if mov == "all":
                for m in ["through", "right", "left"]:
                    green_set.add((app, m))
            else:
                green_set.add((app, mov))

        state = []
        for lnk in links:
            key = (lnk["approach"], lnk["movement"])
            if key in green_set:
                if lnk["movement"] == "through":
                    state.append("G")
                else:
                    state.append("g")
            else:
                state.append("r")
        raw = "".join(state)
        return _sanitize_tls_state_string(raw)

    root = ET.Element("tlLogics")
    tl = ET.SubElement(root, "tlLogic")
    tl.set("id", "J_centre")
    tl.set("type", "static")
    tl.set("programID", "0")
    tl.set("offset", "0")

    for ph in phases:
        state = state_for_phase(ph)
        p = ET.SubElement(tl, "phase")
        p.set("duration", str(ph["duration"]))
        p.set("state", state)
        p.set("name", ph["name"])

    path = os.path.join(out_dir, f"{name}.tll.xml")
    _write_xml(root, path)
    return path


def write_edge_mapping(cfg: dict, out_dir: str) -> str:
    """Write edge_mapping.csv listing all (approach, movement) -> (in_edge, out_edge) pairs."""
    active = cfg["active_approaches"]
    name = cfg["intersection_name"]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    mov_abbr = {"through": "t", "right": "r", "left": "l"}

    rows = []
    for d in active:
        for movement, out_direction in TURN_TARGETS[d].items():
            if out_direction not in active:
                continue
            rows.append({
                "location_slug": slug,
                "approach": d.lower(),
                "movement": mov_abbr[movement],
                "edges": f"edge_{d}_in edge_{out_direction}_out",
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
    print(
        f"  Use with: python preprocessing/data_to_rou.py <csv> "
        f"--net {out_dir}/{name}.net.xml --edge-mapping {path}"
    )
    return path


def run_netconvert(
    cfg: dict, out_dir: str, nod: str, edg: str, con: str, tll: str
) -> str:
    """Run netconvert to produce .net.xml, then post-process phase timing."""
    name = cfg["intersection_name"]
    out_net = os.path.join(out_dir, f"{name}.net.xml")

    min_green = cfg.get("min_green_s", 15)
    max_green = cfg.get("max_green_s", 90)
    amber_s = cfg.get("amber_s", 3)
    mode = str(cfg.get("phases", "2"))

    cmd = [
        "netconvert",
        "--node-files", nod,
        "--edge-files", edg,
        "--connection-files", con,
        "--output-file", out_net,
        "--no-turnarounds", "true",
        "--tls.guess", "true",
        "--tls.cycle.time", str(cfg.get("cycle_s", 120)),
        "--junctions.join", "false",
        "--verbose", "false",
    ]

    print(f"  Running netconvert (auto TLS generation)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("netconvert stderr:", result.stderr[-800:])
        raise RuntimeError("netconvert failed — see above")

    inject_tll_into_net(out_net, tll, tls_id="J_centre")
    _apply_timing_constraints(out_net, min_green, max_green, amber_s, mode)
    write_edge_mapping(cfg, out_dir)

    print(f"  Network written -> {out_net}")
    return out_net


def _apply_timing_constraints(
    net_path: str, min_green: int, max_green: int, amber_s: int, mode: str
):
    """Set long nominal durations for green-only tlLogic; yellow-only rows stay short."""
    _ = min_green, max_green, mode
    long_dur = str(STATIC_GREEN_PHASE_DURATION_S)
    tree = ET.parse(net_path)
    root = tree.getroot()

    for tl in root.findall(".//tlLogic"):
        phases = tl.findall("phase")
        if not phases:
            continue

        for phase in phases:
            state = phase.get("state", "")
            greens = state.count("G") + state.count("g")
            yellows = state.count("y") + state.count("Y")

            if yellows > 0 and greens == 0:
                phase.set("duration", str(amber_s))
            elif greens > 0:
                phase.set("duration", long_dur)
            else:
                phase.set("duration", str(amber_s))

        print(f"  TLS '{tl.get('id')}': {len(phases)} phases adjusted")
        for p in phases:
            print(f"    duration={p.get('duration'):>4s}  state={p.get('state')}")

    tree.write(net_path, encoding="unicode", xml_declaration=True)


def _write_xml(root: ET.Element, path: str):
    raw = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines = [l for l in pretty.split("\n") if not l.startswith("<?xml")]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write("\n".join(lines))
    print(f"  Written -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Build SUMO intersection network from config")
    parser.add_argument("--config", required=True, help="Path to intersection.json")
    parser.add_argument("--out-dir", default="data/sumo/network")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = json.load(fh)

    _root_bn = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root_bn not in sys.path:
        sys.path.insert(0, _root_bn)
    from preprocessing.validate_intersection import validate_and_fix_intersection

    cfg, _warns = validate_and_fix_intersection(cfg, fix=True, log=print)

    name = cfg["intersection_name"]
    print(f"\nBuilding network: {name}")
    print(f"  Approaches : {cfg['active_approaches']}")
    print(f"  Phases     : {cfg.get('phases','2')}-phase")
    print(f"  Left turns : {cfg.get('left_turn_mode', 'permissive')}")
    print(f"  Min green  : {cfg.get('min_green_s',15)}s")
    print(f"  Max green  : {cfg.get('max_green_s',90)}s")
    print(f"  Amber      : {cfg.get('amber_s',3)}s")
    print(f"  Cycle      : {cfg.get('cycle_s',120)}s")
    print()

    os.makedirs(args.out_dir, exist_ok=True)

    nod = write_nodes(cfg, args.out_dir)
    edg = write_edges(cfg, args.out_dir)
    con = write_connections(cfg, args.out_dir)
    tll = write_tll(cfg, args.out_dir)

    try:
        net = run_netconvert(cfg, args.out_dir, nod, edg, con, tll)
        print(f"\nDone. Load in SUMO with: sumo-gui -n {net}")
    except FileNotFoundError:
        print("\nnetconvert not found — intermediate files written successfully.")
        print("Run this command on machine once SUMO is installed:")
        print(
            f"  netconvert --node-files {nod} --edge-files {edg} "
            f"--connection-files {con} "
            f"--output-file {args.out_dir}/{name}.net.xml "
            f"--no-turnarounds true --tls.guess true "
            f"--tls.cycle.time {cfg.get('cycle_s', 120)}"
        )


if __name__ == "__main__":
    main()
