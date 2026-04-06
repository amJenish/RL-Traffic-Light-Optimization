"""
test_BuildNetwork.py
--------------------
Integration + unit tests for BuildNetwork.py using the real project config.

File lives in:  Testcases/test_BuildNetwork.py
Reads from:     ../src/intersection.json
Writes to:      temp directories (never touches src/)

Run from project root:
    python Testcases/test_BuildNetwork.py
    python Testcases/test_BuildNetwork.py -v
"""

import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

TEST_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(TEST_DIR)
SRC_DIR       = os.path.join(PROJECT_ROOT, "src")

PATH_INTERSECTION  = os.path.join(SRC_DIR, "intersection.json")
PATH_BUILDNETWORK  = os.path.join(PROJECT_ROOT, "preprocessing", "BuildNetwork.py")

# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECK
# ---------------------------------------------------------------------------

def _check_required_files():
    missing = []
    for label, path in [
        ("preprocessing/BuildNetwork.py", PATH_BUILDNETWORK),
        ("src/intersection.json",         PATH_INTERSECTION),
    ]:
        if not os.path.exists(path):
            missing.append(f"  MISSING: {label}\n          expected at: {path}")
    if missing:
        print("\nCannot run tests — required files not found:")
        print("\n".join(missing))
        print("\nExpected layout:")
        print("  project/")
        print("  ├── preprocessing/")
        print("  │   └── BuildNetwork.py")
        print("  ├── src/")
        print("  │   └── intersection.json")
        print("  └── Testcases/")
        print("      └── test_BuildNetwork.py")
        sys.exit(1)

_check_required_files()

# ---------------------------------------------------------------------------
# IMPORT BuildNetwork AS A MODULE
# ---------------------------------------------------------------------------

import importlib.util

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bn = _load_module("BuildNetwork", PATH_BUILDNETWORK)

# ---------------------------------------------------------------------------
# LOAD REAL CONFIG ONCE
# ---------------------------------------------------------------------------

with open(PATH_INTERSECTION) as f:
    REAL_CFG = json.load(f)

ACTIVE = REAL_CFG["active_approaches"]
NAME   = REAL_CFG["intersection_name"]

# ---------------------------------------------------------------------------
# FIXTURE HELPERS
# ---------------------------------------------------------------------------

def _make_cfg(active=None, phases="2", min_red=15, amber=3,
              cycle=120, length=200, through=2, right=1, left=1, speed=50):
    active = active or ["N", "S", "E", "W"]
    approaches = {}
    for d in ["N", "S", "E", "W"]:
        approaches[d] = {
            "lanes": {"through": through, "right": right, "left": left},
            "speed_kmh": speed
        }
    return {
        "intersection_name": "Test_Junction",
        "active_approaches": active,
        "approaches": approaches,
        "phases": phases,
        "min_red_s": min_red,
        "amber_s": amber,
        "cycle_s": cycle,
        "edge_length_m": length,
    }


# ===========================================================================
# TEST CLASSES
# ===========================================================================

class TestConfigLoads(unittest.TestCase):
    """Real intersection.json is valid and contains all required keys."""

    def test_required_keys_present(self):
        for key in ["intersection_name", "active_approaches",
                    "approaches", "phases", "min_red_s", "amber_s",
                    "cycle_s", "edge_length_m"]:
            self.assertIn(key, REAL_CFG,
                          f"intersection.json missing key: '{key}'")

    def test_active_approaches_not_empty(self):
        self.assertGreater(len(ACTIVE), 0)

    def test_all_active_approaches_have_lane_config(self):
        for d in ACTIVE:
            self.assertIn(d, REAL_CFG["approaches"],
                          f"Approach '{d}' active but missing from approaches")

    def test_all_approaches_have_lane_counts(self):
        for d in ACTIVE:
            lanes = REAL_CFG["approaches"][d]["lanes"]
            self.assertIn("through", lanes)

    def test_speed_is_positive(self):
        for d in ACTIVE:
            spd = REAL_CFG["approaches"][d]["speed_kmh"]
            self.assertGreater(spd, 0)

    def test_phases_is_valid_value(self):
        self.assertIn(str(REAL_CFG.get("phases", "2")), ["2", "4"])

    def test_min_red_is_positive(self):
        self.assertGreater(REAL_CFG.get("min_red_s", 0), 0)

    def test_cycle_greater_than_amber_and_min_red(self):
        cycle = REAL_CFG.get("cycle_s", 120)
        amber = REAL_CFG.get("amber_s", 3)
        minr  = REAL_CFG.get("min_red_s", 15)
        self.assertGreater(cycle, amber * 2 + minr * 2)


class TestBuildPhases(unittest.TestCase):
    """build_phases() returns green-only service rows (no amber in TLS)."""

    def test_2phase_returns_two_green_bars(self):
        phases = bn.build_phases(["N", "S", "E", "W"], "2", 3, 15, 120)
        self.assertEqual(len(phases), 2)
        names = [p["name"] for p in phases]
        self.assertIn("NS_service", names)
        self.assertIn("EW_service", names)

    def test_4phase_permissive_returns_two_bars(self):
        phases = bn.build_phases(
            ["N", "S", "E", "W"], "4", 3, 15, 120, left_turn_mode="permissive"
        )
        self.assertEqual(len(phases), 2)

    def test_4phase_protected_returns_four_bars_with_lefts(self):
        approaches = {d: {"lanes": {"through": 2, "right": 1, "left": 1}} for d in "NSEW"}
        phases = bn.build_phases(
            ["N", "S", "E", "W"],
            "4",
            3,
            15,
            120,
            approaches,
            left_turn_mode="protected",
        )
        self.assertEqual(len(phases), 4)
        names = [p["name"] for p in phases]
        self.assertIn("NS_left", names)
        self.assertIn("NS_through", names)
        self.assertIn("EW_left", names)
        self.assertIn("EW_through", names)

    def test_all_phases_have_required_keys(self):
        for mode in ["2", "4"]:
            phases = bn.build_phases(
                ["N", "S", "E", "W"], mode, 3, 15, 120, left_turn_mode="permissive"
            )
            for ph in phases:
                self.assertIn("name", ph)
                self.assertIn("duration", ph)
                self.assertIn("green_groups", ph)

    def test_all_durations_positive(self):
        for mode in ["2", "4"]:
            phases = bn.build_phases(
                ["N", "S", "E", "W"], mode, 3, 15, 120, left_turn_mode="permissive"
            )
            for ph in phases:
                self.assertGreater(ph["duration"], 0,
                                   f"Phase '{ph['name']}' has zero duration")

    def test_no_amber_rows_in_program(self):
        phases = bn.build_phases(
            ["N", "S", "E", "W"], "2", 3, 15, 120, left_turn_mode="permissive"
        )
        for ph in phases:
            self.assertNotIn("amber", ph["name"].lower())
            self.assertGreater(len(ph["green_groups"]), 0)

    def test_4phase_protected_no_left_lanes_skips_left_bars(self):
        approaches = {d: {"lanes": {"through": 2, "right": 1, "left": 0}} for d in "NSEW"}
        phases = bn.build_phases(
            ["N", "S", "E", "W"],
            "4",
            3,
            15,
            120,
            approaches,
            left_turn_mode="protected",
        )
        names = [p["name"] for p in phases]
        self.assertNotIn("NS_left", names)
        self.assertNotIn("EW_left", names)
        self.assertEqual(len(phases), 2)
        self.assertIn("NS_through", names)
        self.assertIn("EW_through", names)

    def test_static_green_duration_large(self):
        phases = bn.build_phases(
            ["N", "S", "E", "W"], "2", 3, 15, 30, left_turn_mode="permissive"
        )
        for ph in phases:
            self.assertGreaterEqual(ph["duration"], bn.STATIC_GREEN_PHASE_DURATION_S)

    def test_real_config_produces_valid_phases(self):
        phases = bn.build_phases(
            ACTIVE,
            str(REAL_CFG.get("phases", "2")),
            REAL_CFG.get("amber_s", 3),
            REAL_CFG.get("min_red_s", 15),
            REAL_CFG.get("cycle_s", 120),
            left_turn_mode=str(REAL_CFG.get("left_turn_mode", "permissive")),
        )
        self.assertGreater(len(phases), 0)
        for ph in phases:
            self.assertGreater(ph["duration"], 0)


class TestWriteNodes(unittest.TestCase):
    """write_nodes() produces valid .nod.xml with correct structure."""

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        cls.path = bn.write_nodes(REAL_CFG, cls.tmp)
        cls.root = ET.parse(cls.path).getroot()

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.path))

    def test_filename_matches_intersection_name(self):
        self.assertEqual(os.path.basename(self.path), f"{NAME}.nod.xml")

    def test_root_tag_is_nodes(self):
        self.assertEqual(self.root.tag, "nodes")

    def test_centre_junction_exists(self):
        ids = [n.get("id") for n in self.root.findall("node")]
        self.assertIn("J_centre", ids)

    def test_centre_junction_is_traffic_light(self):
        centre = next(n for n in self.root.findall("node")
                      if n.get("id") == "J_centre")
        self.assertEqual(centre.get("type"), "traffic_light")

    def test_centre_junction_at_origin(self):
        centre = next(n for n in self.root.findall("node")
                      if n.get("id") == "J_centre")
        self.assertEqual(centre.get("x"), "0")
        self.assertEqual(centre.get("y"), "0")

    def test_approach_nodes_created_for_all_active(self):
        ids = [n.get("id") for n in self.root.findall("node")]
        for d in ACTIVE:
            self.assertIn(f"node_{d}_in", ids,
                          f"node_{d}_in missing from .nod.xml")

    def test_inactive_approaches_not_present(self):
        ids      = [n.get("id") for n in self.root.findall("node")]
        inactive = [d for d in ["N","S","E","W"] if d not in ACTIVE]
        for d in inactive:
            self.assertNotIn(f"node_{d}_in", ids)

    def test_approach_node_positions_match_direction_vectors(self):
        length = REAL_CFG.get("edge_length_m", 200)
        for d in ACTIVE:
            node = next(n for n in self.root.findall("node")
                        if n.get("id") == f"node_{d}_in")
            dx, dy = bn.DIRECTION_VECTORS[d]
            self.assertEqual(node.get("x"), str(dx * length))
            self.assertEqual(node.get("y"), str(dy * length))

    def test_total_node_count(self):
        # 1 centre + 1 per active approach
        expected = 1 + len(ACTIVE)
        self.assertEqual(len(self.root.findall("node")), expected)

    def test_valid_xml(self):
        try:
            ET.parse(self.path)
        except ET.ParseError as e:
            self.fail(f".nod.xml is not valid XML: {e}")


class TestWriteEdges(unittest.TestCase):
    """write_edges() produces valid .edg.xml with correct lane counts and speeds."""

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        cls.path = bn.write_edges(REAL_CFG, cls.tmp)
        cls.root = ET.parse(cls.path).getroot()

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.path))

    def test_filename_matches_intersection_name(self):
        self.assertEqual(os.path.basename(self.path), f"{NAME}.edg.xml")

    def test_root_tag_is_edges(self):
        self.assertEqual(self.root.tag, "edges")

    def test_inbound_and_outbound_edges_for_all_active(self):
        ids = [e.get("id") for e in self.root.findall("edge")]
        for d in ACTIVE:
            self.assertIn(f"edge_{d}_in",  ids)
            self.assertIn(f"edge_{d}_out", ids)

    def test_inactive_approaches_have_no_edges(self):
        ids      = [e.get("id") for e in self.root.findall("edge")]
        inactive = [d for d in ["N","S","E","W"] if d not in ACTIVE]
        for d in inactive:
            self.assertNotIn(f"edge_{d}_in",  ids)
            self.assertNotIn(f"edge_{d}_out", ids)

    def test_total_edge_count(self):
        # 2 edges (in + out) per active approach
        self.assertEqual(len(self.root.findall("edge")), len(ACTIVE) * 2)

    def test_inbound_edges_go_to_centre(self):
        for d in ACTIVE:
            edge = next(e for e in self.root.findall("edge")
                        if e.get("id") == f"edge_{d}_in")
            self.assertEqual(edge.get("to"), "J_centre")

    def test_outbound_edges_come_from_centre(self):
        for d in ACTIVE:
            edge = next(e for e in self.root.findall("edge")
                        if e.get("id") == f"edge_{d}_out")
            self.assertEqual(edge.get("from"), "J_centre")

    def test_inbound_lane_count_matches_config(self):
        for d in ACTIVE:
            expected = bn._total_lanes(REAL_CFG["approaches"][d])
            edge     = next(e for e in self.root.findall("edge")
                            if e.get("id") == f"edge_{d}_in")
            self.assertEqual(int(edge.get("numLanes")), expected,
                             f"edge_{d}_in lane count mismatch")

    def test_speed_converted_from_kmh_to_ms(self):
        for d in ACTIVE:
            spd_kmh  = REAL_CFG["approaches"][d]["speed_kmh"]
            expected = round(spd_kmh / 3.6, 2)
            edge     = next(e for e in self.root.findall("edge")
                            if e.get("id") == f"edge_{d}_in")
            self.assertAlmostEqual(float(edge.get("speed")), expected, places=1)

    def test_edge_length_matches_config(self):
        length = str(REAL_CFG.get("edge_length_m", 200))
        for d in ACTIVE:
            for suffix in ["_in", "_out"]:
                edge = next(e for e in self.root.findall("edge")
                            if e.get("id") == f"edge_{d}{suffix}")
                self.assertEqual(edge.get("length"), length)

    def test_valid_xml(self):
        try:
            ET.parse(self.path)
        except ET.ParseError as e:
            self.fail(f".edg.xml is not valid XML: {e}")


class TestWriteConnections(unittest.TestCase):
    """write_connections() produces valid .con.xml with correct turn mappings."""

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        cls.path = bn.write_connections(REAL_CFG, cls.tmp)
        cls.root = ET.parse(cls.path).getroot()

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.path))

    def test_filename_matches_intersection_name(self):
        self.assertEqual(os.path.basename(self.path), f"{NAME}.con.xml")

    def test_root_tag_is_connections(self):
        self.assertEqual(self.root.tag, "connections")

    def test_connections_exist(self):
        self.assertGreater(len(self.root.findall("connection")), 0)

    def test_all_from_edges_are_inbound(self):
        for con in self.root.findall("connection"):
            self.assertTrue(con.get("from").endswith("_in"),
                            f"Connection 'from' not an inbound edge: {con.get('from')}")

    def test_all_to_edges_are_outbound(self):
        for con in self.root.findall("connection"):
            self.assertTrue(con.get("to").endswith("_out"),
                            f"Connection 'to' not an outbound edge: {con.get('to')}")

    def test_from_edges_only_reference_active_approaches(self):
        valid = {f"edge_{d}_in" for d in ACTIVE}
        for con in self.root.findall("connection"):
            self.assertIn(con.get("from"), valid)

    def test_to_edges_only_reference_active_approaches(self):
        valid = {f"edge_{d}_out" for d in ACTIVE}
        for con in self.root.findall("connection"):
            self.assertIn(con.get("to"), valid)

    def test_lane_indices_are_non_negative(self):
        for con in self.root.findall("connection"):
            self.assertGreaterEqual(int(con.get("fromLane")), 0)
            self.assertGreaterEqual(int(con.get("toLane")),   0)

    def test_turn_targets_follow_right_hand_traffic(self):
        # From North: right → East, through → South, left → West
        n_cons = [c for c in self.root.findall("connection")
                  if c.get("from") == "edge_N_in"]
        if n_cons:
            targets = {c.get("to") for c in n_cons}
            # At least one connection should go toward South (through)
            if "S" in ACTIVE:
                self.assertIn("edge_S_out", targets)

    def test_t_intersection_omits_missing_direction(self):
        """If W is inactive, no connections should reference edge_W_out."""
        if "W" not in ACTIVE:
            for con in self.root.findall("connection"):
                self.assertNotEqual(con.get("to"), "edge_W_out")

    def test_valid_xml(self):
        try:
            ET.parse(self.path)
        except ET.ParseError as e:
            self.fail(f".con.xml is not valid XML: {e}")


class TestWriteTll(unittest.TestCase):
    """write_tll() produces valid .tll.xml with correct signal logic."""

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        cls.path = bn.write_tll(REAL_CFG, cls.tmp)
        cls.root = ET.parse(cls.path).getroot()
        cls.tl   = cls.root.find("tlLogic")
        cls.mode = str(REAL_CFG.get("phases", "2"))

    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.path))

    def test_filename_matches_intersection_name(self):
        self.assertEqual(os.path.basename(self.path), f"{NAME}.tll.xml")

    def test_root_tag_is_tllogics(self):
        self.assertEqual(self.root.tag, "tlLogics")

    def test_tllogic_element_present(self):
        self.assertIsNotNone(self.tl, "No tlLogic element found")

    def test_tllogic_id_is_j_centre(self):
        self.assertEqual(self.tl.get("id"), "J_centre")

    def test_tllogic_type_is_static(self):
        self.assertEqual(self.tl.get("type"), "static")

    def test_correct_number_of_phases(self):
        phases = self.tl.findall("phase")
        lt = str(REAL_CFG.get("left_turn_mode", "permissive")).lower()
        if self.mode == "2" or lt == "permissive":
            expected = 2
        else:
            expected = 4
        self.assertEqual(len(phases), expected)

    def test_all_phases_have_duration(self):
        for ph in self.tl.findall("phase"):
            dur = ph.get("duration")
            self.assertIsNotNone(dur)
            self.assertGreater(int(dur), 0)

    def test_all_phases_have_state_string(self):
        for ph in self.tl.findall("phase"):
            state = ph.get("state")
            self.assertIsNotNone(state)
            self.assertGreater(len(state), 0)

    def test_state_strings_all_same_length(self):
        phases  = self.tl.findall("phase")
        lengths = {len(ph.get("state", "")) for ph in phases}
        self.assertEqual(len(lengths), 1,
                         f"State strings have inconsistent lengths: {lengths}")

    def test_state_strings_only_contain_valid_chars(self):
        valid = set("GgrR")
        for ph in self.tl.findall("phase"):
            state = ph.get("state", "")
            bad = set(state) - valid
            self.assertEqual(bad, set(),
                             f"Invalid chars in state '{state}': {bad}")

    def test_at_least_one_green_phase(self):
        phases = self.tl.findall("phase")
        has_green = any("G" in ph.get("state", "") for ph in phases)
        self.assertTrue(has_green, "No phase has any green signal")

    def test_no_yellow_in_tls_program_states(self):
        for ph in self.tl.findall("phase"):
            state = ph.get("state", "")
            self.assertTrue(
                "y" not in state and "Y" not in state,
                f"Phase '{ph.get('name')}' must be green-only in TLS",
            )

    def test_phase_names_present(self):
        for ph in self.tl.findall("phase"):
            self.assertIsNotNone(ph.get("name"))

    def test_ns_ew_states_match_sumo_nesw_link_order(self):
        """SUMO incLanes order is N(0-3), E(4-7), S(8-11), W(12-15)."""
        phases = self.tl.findall("phase")
        if len(phases) != 2:
            self.skipTest("NESW slot test applies to 2-bar permissive configs")
        ns = next(p for p in phases if p.get("name") == "NS_service")
        ew = next(p for p in phases if p.get("name") == "EW_service")
        st_ns = ns.get("state", "")
        st_ew = ew.get("state", "")
        self.assertEqual(len(st_ns), 16)
        self.assertEqual(len(st_ew), 16)
        for i in range(4):
            self.assertIn(st_ns[i], "Gg")
        for i in range(4, 8):
            self.assertEqual(st_ns[i], "r")
        for i in range(8, 12):
            self.assertIn(st_ns[i], "Gg")
        for i in range(12, 16):
            self.assertEqual(st_ns[i], "r")
        for i in range(4):
            self.assertEqual(st_ew[i], "r")
        for i in range(4, 8):
            self.assertIn(st_ew[i], "Gg")
        for i in range(8, 12):
            self.assertEqual(st_ew[i], "r")
        for i in range(12, 16):
            self.assertIn(st_ew[i], "Gg")

    def test_valid_xml(self):
        try:
            ET.parse(self.path)
        except ET.ParseError as e:
            self.fail(f".tll.xml is not valid XML: {e}")


class TestTotalLanes(unittest.TestCase):
    """_total_lanes() computes correct sum from lane config."""

    def test_through_right_left(self):
        cfg = {"lanes": {"through": 2, "right": 1, "left": 1}}
        self.assertEqual(bn._total_lanes(cfg), 4)

    def test_through_only(self):
        cfg = {"lanes": {"through": 3}}
        self.assertEqual(bn._total_lanes(cfg), 3)

    def test_through_and_right_no_left(self):
        cfg = {"lanes": {"through": 2, "right": 1, "left": 0}}
        self.assertEqual(bn._total_lanes(cfg), 3)

    def test_single_lane(self):
        cfg = {"lanes": {"through": 1, "right": 0, "left": 0}}
        self.assertEqual(bn._total_lanes(cfg), 1)

    def test_real_config_values(self):
        for d in ACTIVE:
            app      = REAL_CFG["approaches"][d]
            lanes    = app["lanes"]
            expected = lanes.get("through",0) + lanes.get("right",0) + lanes.get("left",0)
            self.assertEqual(bn._total_lanes(app), expected)


class TestAllFilesWrittenTogether(unittest.TestCase):
    """All four intermediate files are produced correctly in one output dir."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.nod = bn.write_nodes(REAL_CFG,       cls.tmp)
        cls.edg = bn.write_edges(REAL_CFG,       cls.tmp)
        cls.con = bn.write_connections(REAL_CFG, cls.tmp)
        cls.tll = bn.write_tll(REAL_CFG,         cls.tmp)

    def test_all_four_files_exist(self):
        for path in [self.nod, self.edg, self.con, self.tll]:
            self.assertTrue(os.path.exists(path), f"Missing: {path}")

    def test_all_four_are_valid_xml(self):
        for path in [self.nod, self.edg, self.con, self.tll]:
            try:
                ET.parse(path)
            except ET.ParseError as e:
                self.fail(f"{os.path.basename(path)} is not valid XML: {e}")

    def test_files_written_to_correct_directory(self):
        for path in [self.nod, self.edg, self.con, self.tll]:
            self.assertEqual(os.path.dirname(path), self.tmp)

    def test_edge_ids_consistent_across_files(self):
        """Edge IDs in .edg.xml must match what .con.xml references."""
        edg_ids  = {e.get("id") for e in ET.parse(self.edg).getroot().findall("edge")}
        con_from = {c.get("from") for c in ET.parse(self.con).getroot().findall("connection")}
        con_to   = {c.get("to")   for c in ET.parse(self.con).getroot().findall("connection")}
        for eid in con_from | con_to:
            self.assertIn(eid, edg_ids,
                          f"Connection references '{eid}' not in .edg.xml")

    def test_node_ids_consistent_across_files(self):
        """Node IDs in .nod.xml must match from/to in .edg.xml."""
        nod_ids  = {n.get("id") for n in ET.parse(self.nod).getroot().findall("node")}
        edg_root = ET.parse(self.edg).getroot()
        for e in edg_root.findall("edge"):
            self.assertIn(e.get("from"), nod_ids,
                          f"Edge '{e.get('id')}' from='{e.get('from')}' not in .nod.xml")
            self.assertIn(e.get("to"), nod_ids,
                          f"Edge '{e.get('id')}' to='{e.get('to')}' not in .nod.xml")


class TestEdgeCases(unittest.TestCase):
    """Non-standard configurations produce correct output."""

    def test_3_approach_t_intersection(self):
        cfg = _make_cfg(active=["N", "S", "E"])
        tmp = tempfile.mkdtemp()
        bn.write_nodes(cfg, tmp)
        bn.write_edges(cfg, tmp)
        bn.write_connections(cfg, tmp)
        bn.write_tll(cfg, tmp)
        # West should not appear in any file
        for fname in os.listdir(tmp):
            root = ET.parse(os.path.join(tmp, fname)).getroot()
            xml_str = ET.tostring(root, encoding="unicode")
            self.assertNotIn("edge_W_in",  xml_str,
                             f"{fname} references inactive West approach")
            self.assertNotIn("node_W_in",  xml_str)

    def test_4_phase_mode(self):
        cfg = _make_cfg(phases="4")
        tmp = tempfile.mkdtemp()
        path = bn.write_tll(cfg, tmp)
        root = ET.parse(path).getroot()
        phases = root.find("tlLogic").findall("phase")
        self.assertEqual(len(phases), 2)

    def test_through_only_lanes(self):
        cfg = _make_cfg(through=2, right=0, left=0)
        tmp = tempfile.mkdtemp()
        bn.write_nodes(cfg, tmp)
        path = bn.write_edges(cfg, tmp)
        root = ET.parse(path).getroot()
        for d in ["N", "S", "E", "W"]:
            edge = next(e for e in root.findall("edge")
                        if e.get("id") == f"edge_{d}_in")
            self.assertEqual(int(edge.get("numLanes")), 2)

    def test_custom_edge_length(self):
        cfg  = _make_cfg(length=500)
        tmp  = tempfile.mkdtemp()
        path = bn.write_edges(cfg, tmp)
        root = ET.parse(path).getroot()
        for e in root.findall("edge"):
            self.assertEqual(e.get("length"), "500")

    def test_custom_speed(self):
        cfg      = _make_cfg(speed=80)
        tmp      = tempfile.mkdtemp()
        path     = bn.write_edges(cfg, tmp)
        root     = ET.parse(path).getroot()
        expected = round(80 / 3.6, 2)
        for e in root.findall("edge"):
            self.assertAlmostEqual(float(e.get("speed")), expected, places=1)

    def test_netconvert_skipped_gracefully_if_not_installed(self):
        """run_netconvert raises FileNotFoundError if netconvert absent — not a crash."""
        import subprocess
        import unittest.mock as mock
        cfg = _make_cfg()
        tmp = tempfile.mkdtemp()
        nod = bn.write_nodes(cfg, tmp)
        edg = bn.write_edges(cfg, tmp)
        con = bn.write_connections(cfg, tmp)
        tll = bn.write_tll(cfg, tmp)
        with mock.patch("subprocess.run",
                        side_effect=FileNotFoundError("netconvert not found")):
            with self.assertRaises(FileNotFoundError):
                bn.run_netconvert(cfg, tmp, nod, edg, con, tll)


class TestInjectTllIntoNet(unittest.TestCase):
    """inject_tll_into_net must place tlLogic before <connection> (SUMO load order)."""

    def test_tl_logic_before_connections(self):
        net_xml = """<?xml version='1.0' encoding='UTF-8'?>
<net version="1.20">
  <location netOffset="0,0" convBoundary="0,0,1,1" origBoundary="0,0,1,1" projParameter="!"/>
  <junction id="J_centre" type="traffic_light" x="0" y="0" incLanes="" intLanes="" shape="0,0"/>
  <connection from="e1" to="e2" fromLane="0" toLane="0" tl="J_centre" linkIndex="0" dir="s" state="M"/>
  <tlLogic id="J_centre" type="static" programID="0" offset="0">
    <phase duration="30" state="G"/>
  </tlLogic>
</net>
"""
        tll_xml = """<?xml version='1.0' encoding='UTF-8'?>
<tlLogics>
  <tlLogic id="J_centre" type="static" programID="0" offset="0">
    <phase duration="999" state="r"/>
  </tlLogic>
</tlLogics>
"""
        with tempfile.TemporaryDirectory() as tmp:
            net_path = os.path.join(tmp, "test.net.xml")
            tll_path = os.path.join(tmp, "test.tll.xml")
            with open(net_path, "w", encoding="utf-8") as f:
                f.write(net_xml)
            with open(tll_path, "w", encoding="utf-8") as f:
                f.write(tll_xml)
            bn.inject_tll_into_net(net_path, tll_path, tls_id="J_centre")
            root = ET.parse(net_path).getroot()
            tags = [c.tag for c in root]
            self.assertLess(tags.index("tlLogic"), tags.index("connection"))
            phase = root.find("tlLogic").find("phase")
            self.assertEqual(phase.get("duration"), "999")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)