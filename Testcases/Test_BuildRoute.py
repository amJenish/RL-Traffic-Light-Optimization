"""
test_BuildRoute.py
------------------
Integration + unit tests for BuildRoute.py using the real project data.

File lives in:  preprocessing/test_BuildRoute.py
Reads from:     ../src/intersection.json
                ../src/columns.json
                ../src/data/synthetic_toronto_data.xls
Writes to:      temp directories (never touches src/)

Run from project root:
    python preprocessing/test_BuildRoute.py
    python preprocessing/test_BuildRoute.py -v

Run from inside preprocessing/:
    python test_BuildRoute.py
    python test_BuildRoute.py -v
"""

import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

TEST_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")

PATH_INTERSECTION = os.path.join(SRC_DIR, "intersection.json")
PATH_COLUMNS      = os.path.join(SRC_DIR, "columns.json")
PATH_CSV          = os.path.join(SRC_DIR, "data", "synthetic_toronto_data.xls")
PATH_BUILDROUTE   = os.path.join(PROJECT_ROOT, "preprocessing", "BuildRoute.py")

# ---------------------------------------------------------------------------
# PRE-FLIGHT CHECK
# ---------------------------------------------------------------------------

def _check_required_files():
    missing = []
    for label, path in [
        ("preprocessing/BuildRoute.py",              PATH_BUILDROUTE),
        ("src/intersection.json",                    PATH_INTERSECTION),
        ("src/columns.json",                         PATH_COLUMNS),
        ("src/data/synthetic_toronto_data.xls",      PATH_CSV),
    ]:
        if not os.path.exists(path):
            missing.append(f"  MISSING: {label}\n          expected at: {path}")
    if missing:
        print("\nCannot run tests — required files not found:")
        print("\n".join(missing))
        print("\nExpected layout:")
        print("  project/")
        print("  ├── preprocessing/")
        print("  │   └── BuildRoute.py")
        print("  ├── src/")
        print("  │   ├── intersection.json")
        print("  │   ├── columns.json")
        print("  │   └── data/")
        print("  │       └── synthetic_toronto_data.xls")
        print("  └── Testcases/")
        print("      └── test_BuildRoute.py")
        sys.exit(1)

_check_required_files()

# ---------------------------------------------------------------------------
# IMPORT BuildRoute
# ---------------------------------------------------------------------------

import importlib.util

def _load_module(name, filepath):
    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

br = _load_module("BuildRoute", PATH_BUILDROUTE)

# ---------------------------------------------------------------------------
# LOAD REAL CONFIGS ONCE
# ---------------------------------------------------------------------------

with open(PATH_INTERSECTION) as f:
    REAL_INTERSECTION = json.load(f)

with open(PATH_COLUMNS) as f:
    REAL_COLUMNS = json.load(f)

ACTIVE    = REAL_INTERSECTION["active_approaches"]
SLOT_MINS = REAL_COLUMNS.get("slot_minutes", 15)

N_DAYS  = 60
N_SLOTS = 32
N_TRAIN = 54
N_TEST  = 6

EXPECTED_FIRST_SLOT = "07:30:00"
EXPECTED_LAST_SLOT  = "17:45:00"
EXPECTED_DATA_START = 7 * 3600 + 30 * 60   # 27000s


def _run_pipeline(tmp_dir, test_days=N_TEST, seed=42):
    sys.argv = [
        "BuildRoute.py",
        "--csv",          PATH_CSV,
        "--intersection", PATH_INTERSECTION,
        "--columns",      PATH_COLUMNS,
        "--out-dir",      tmp_dir,
        "--test-days",    str(test_days),
        "--seed",         str(seed),
    ]
    br.main()
    return {
        "days_dir":   os.path.join(tmp_dir, "processed", "days"),
        "flows_dir":  os.path.join(tmp_dir, "sumo", "flows"),
        "split_path": os.path.join(tmp_dir, "processed", "split.json"),
    }


# ===========================================================================

class TestConfigsLoad(unittest.TestCase):

    def test_intersection_has_required_keys(self):
        for key in ["intersection_name", "active_approaches", "approaches", "min_red_s"]:
            self.assertIn(key, REAL_INTERSECTION)

    def test_active_approaches_not_empty(self):
        self.assertGreater(len(ACTIVE), 0)

    def test_all_active_approaches_have_lane_config(self):
        for d in ACTIVE:
            self.assertIn(d, REAL_INTERSECTION["approaches"])

    def test_columns_has_required_keys(self):
        self.assertIn("time",       REAL_COLUMNS)
        self.assertIn("approaches", REAL_COLUMNS)
        self.assertIn("start_time", REAL_COLUMNS["time"])

    def test_slot_minutes_is_positive(self):
        self.assertGreater(SLOT_MINS, 0)

    def test_min_red_is_positive(self):
        self.assertGreater(REAL_INTERSECTION.get("min_red_s", 0), 0)

    def test_all_active_approaches_mapped_in_columns(self):
        for d in ACTIVE:
            self.assertIn(d, REAL_COLUMNS["approaches"])


class TestCsvLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = br.load_csv(PATH_CSV, REAL_COLUMNS, ACTIVE)

    def test_start_time_column_exists(self):
        self.assertIn("start_time", self.df.columns)

    def test_correct_total_rows(self):
        self.assertEqual(len(self.df), N_DAYS * N_SLOTS)

    def test_totals_computed_for_all_approaches(self):
        for d in ACTIVE:
            self.assertIn(f"{d}_total", self.df.columns)

    def test_no_pedestrian_columns(self):
        peds = [c for c in self.df.columns if "ped" in c.lower()]
        self.assertEqual(peds, [])

    def test_no_negative_counts(self):
        for d in ACTIVE:
            for m in ["through", "right", "left", "total"]:
                col = f"{d}_{m}"
                if col in self.df.columns:
                    self.assertTrue((self.df[col] >= 0).all())

    def test_total_equals_sum_of_movements(self):
        for d in ACTIVE:
            expected = self.df[f"{d}_through"] + self.df[f"{d}_right"] + self.df[f"{d}_left"]
            pd.testing.assert_series_equal(
                self.df[f"{d}_total"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False
            )

    def test_correct_number_of_unique_slots(self):
        self.assertEqual(self.df["start_time"].nunique(), N_SLOTS)

    def test_first_slot_is_07_30(self):
        self.assertEqual(sorted(self.df["start_time"].unique())[0], EXPECTED_FIRST_SLOT)

    def test_last_slot_is_17_45(self):
        self.assertEqual(sorted(self.df["start_time"].unique())[-1], EXPECTED_LAST_SLOT)


class TestSimDayAssignment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raw    = br.load_csv(PATH_CSV, REAL_COLUMNS, ACTIVE)
        cls.df = br.assign_sim_days(raw)

    def test_correct_number_of_days(self):
        self.assertEqual(self.df["sim_day"].nunique(), N_DAYS)

    def test_day_range_zero_to_59(self):
        self.assertEqual(self.df["sim_day"].min(), 0)
        self.assertEqual(self.df["sim_day"].max(), N_DAYS - 1)

    def test_each_day_has_32_slots(self):
        counts = self.df.groupby("sim_day").size()
        self.assertTrue((counts == N_SLOTS).all())

    def test_slots_sorted_within_each_day(self):
        for day_id in [0, 1, 30, 59]:
            times = self.df[self.df["sim_day"] == day_id]["start_time"].tolist()
            self.assertEqual(times, sorted(times))

    def test_days_have_different_volumes(self):
        slot_df = self.df[self.df["start_time"] == "08:30:00"]
        vals    = slot_df[f"{ACTIVE[0]}_total"].values
        self.assertGreater(vals.max() - vals.min(), 0)


class TestDayCSVFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp      = tempfile.mkdtemp()
        paths        = _run_pipeline(cls.tmp)
        cls.days_dir = paths["days_dir"]

    def test_60_files_written(self):
        self.assertEqual(len(os.listdir(self.days_dir)), N_DAYS)

    def test_all_files_named_correctly(self):
        for i in range(N_DAYS):
            self.assertTrue(
                os.path.exists(os.path.join(self.days_dir, f"day_{i:02d}.csv"))
            )

    def test_each_file_has_32_rows(self):
        for i in [0, 1, 30, 59]:
            df = pd.read_csv(os.path.join(self.days_dir, f"day_{i:02d}.csv"))
            self.assertEqual(len(df), N_SLOTS)

    def test_required_columns_present(self):
        df = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        self.assertIn("sim_day",    df.columns)
        self.assertIn("start_time", df.columns)
        for d in ACTIVE:
            self.assertIn(f"{d}_total",   df.columns)
            self.assertIn(f"{d}_through", df.columns)

    def test_no_peds_in_output(self):
        df   = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        peds = [c for c in df.columns if "ped" in c.lower()]
        self.assertEqual(peds, [])

    def test_sim_day_matches_filename(self):
        for i in [0, 15, 59]:
            df = pd.read_csv(os.path.join(self.days_dir, f"day_{i:02d}.csv"))
            self.assertTrue((df["sim_day"] == i).all())

    def test_first_slot_is_07_30(self):
        df = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        self.assertEqual(df["start_time"].iloc[0], EXPECTED_FIRST_SLOT)

    def test_last_slot_is_17_45(self):
        df = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        self.assertEqual(df["start_time"].iloc[-1], EXPECTED_LAST_SLOT)

    def test_total_equals_sum_of_movements(self):
        df = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        for d in ACTIVE:
            expected = df[f"{d}_through"] + df[f"{d}_right"] + df[f"{d}_left"]
            pd.testing.assert_series_equal(
                df[f"{d}_total"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False
            )

    def test_different_days_have_different_volumes(self):
        df0 = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        df1 = pd.read_csv(os.path.join(self.days_dir, "day_01.csv"))
        self.assertFalse(df0[f"{ACTIVE[0]}_total"].equals(df1[f"{ACTIVE[0]}_total"]))

    def test_no_zero_totals_during_peak(self):
        df   = pd.read_csv(os.path.join(self.days_dir, "day_00.csv"))
        peak = df[df["start_time"].isin(["08:15:00", "08:30:00", "08:45:00"])]
        for d in ACTIVE:
            self.assertTrue((peak[f"{d}_total"] > 0).all())


class TestSumoFlowFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp       = tempfile.mkdtemp()
        paths         = _run_pipeline(cls.tmp)
        cls.flows_dir = paths["flows_dir"]

    def test_60_flow_files_written(self):
        files = [f for f in os.listdir(self.flows_dir) if f.endswith(".rou.xml")]
        self.assertEqual(len(files), N_DAYS)

    def test_all_files_valid_xml(self):
        for i in range(N_DAYS):
            path = os.path.join(self.flows_dir, f"flows_day_{i:02d}.rou.xml")
            try:
                ET.parse(path)
            except ET.ParseError as e:
                self.fail(f"flows_day_{i:02d}.rou.xml invalid XML: {e}")

    def test_root_tag_is_routes(self):
        root = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        self.assertEqual(root.tag, "routes")

    def test_vtype_passenger_present(self):
        root   = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        vtypes = [v for v in root.findall("vType") if v.get("id") == "passenger"]
        self.assertEqual(len(vtypes), 1)

    def test_edge_ids_follow_convention(self):
        root       = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        expected   = {br.edge_in_id(d) for d in ACTIVE}
        actual     = {f.get("from") for f in root.findall("flow")}
        unexpected = actual - expected
        self.assertEqual(unexpected, set())

    def test_all_active_approaches_have_flows(self):
        root = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        used = {f.get("from") for f in root.findall("flow")}
        for d in ACTIVE:
            self.assertIn(br.edge_in_id(d), used)

    def test_overnight_flows_present(self):
        root      = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        overnight = [f for f in root.findall("flow") if f.get("begin") == "0"]
        self.assertGreater(len(overnight), 0)

    def test_data_window_flows_at_07_30(self):
        root       = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        data_flows = [f for f in root.findall("flow")
                      if f.get("begin") == str(EXPECTED_DATA_START)]
        self.assertGreater(len(data_flows), 0)

    def test_no_overlapping_flows_per_approach(self):
        root = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        for d in ACTIVE:
            edge  = br.edge_in_id(d)
            flows = sorted([(int(f.get("begin")), int(f.get("end")))
                             for f in root.findall("flow") if f.get("from") == edge])
            for i in range(len(flows) - 1):
                self.assertLessEqual(flows[i][1], flows[i+1][0])

    def test_all_vph_positive(self):
        root = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        for f in root.findall("flow"):
            self.assertGreater(float(f.get("vehsPerHour", 0)), 0)

    def test_peak_vph_higher_than_overnight(self):
        root     = ET.parse(os.path.join(self.flows_dir, "flows_day_00.rou.xml")).getroot()
        edge     = br.edge_in_id(ACTIVE[0])
        ov_vph   = [float(f.get("vehsPerHour")) for f in root.findall("flow")
                    if f.get("begin") == "0" and f.get("from") == edge]
        pk_begin = str(8 * 3600 + 30 * 60)
        pk_vph   = [float(f.get("vehsPerHour")) for f in root.findall("flow")
                    if f.get("begin") == pk_begin and f.get("from") == edge]
        if ov_vph and pk_vph:
            self.assertGreater(pk_vph[0], ov_vph[0])

    def test_different_days_have_different_peak_flows(self):
        def get_vph(day_id):
            root  = ET.parse(os.path.join(
                self.flows_dir, f"flows_day_{day_id:02d}.rou.xml")).getroot()
            flows = [f for f in root.findall("flow")
                     if f.get("begin") == str(EXPECTED_DATA_START)
                     and f.get("from") == br.edge_in_id(ACTIVE[0])]
            return float(flows[0].get("vehsPerHour")) if flows else None
        self.assertNotEqual(get_vph(0), get_vph(1))


class TestSplitJson(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        paths    = _run_pipeline(cls.tmp)
        with open(paths["split_path"]) as f:
            cls.split = json.load(f)

    def test_all_required_keys_present(self):
        for key in ["train", "test", "intersection_name", "active_approaches",
                    "min_red_seconds", "n_days", "n_slots", "slot_minutes", "edge_ids"]:
            self.assertIn(key, self.split)

    def test_n_days_is_60(self):
        self.assertEqual(self.split["n_days"], N_DAYS)

    def test_n_slots_is_32(self):
        self.assertEqual(self.split["n_slots"], N_SLOTS)

    def test_slot_minutes_is_15(self):
        self.assertEqual(self.split["slot_minutes"], 15)

    def test_train_count_is_54(self):
        self.assertEqual(len(self.split["train"]), N_TRAIN)

    def test_test_count_is_6(self):
        self.assertEqual(len(self.split["test"]), N_TEST)

    def test_no_overlap(self):
        overlap = set(self.split["train"]) & set(self.split["test"])
        self.assertEqual(overlap, set())

    def test_all_days_accounted_for(self):
        all_days = set(self.split["train"]) | set(self.split["test"])
        self.assertEqual(all_days, set(range(N_DAYS)))

    def test_intersection_name_matches_config(self):
        self.assertEqual(self.split["intersection_name"],
                         REAL_INTERSECTION["intersection_name"])

    def test_active_approaches_match_config(self):
        self.assertEqual(sorted(self.split["active_approaches"]), sorted(ACTIVE))

    def test_min_red_matches_config(self):
        self.assertEqual(self.split["min_red_seconds"],
                         REAL_INTERSECTION.get("min_red_s", 15))

    def test_edge_ids_follow_convention(self):
        for d in ACTIVE:
            self.assertIn(d, self.split["edge_ids"])
            self.assertEqual(self.split["edge_ids"][d]["in"],  f"edge_{d}_in")
            self.assertEqual(self.split["edge_ids"][d]["out"], f"edge_{d}_out")

    def test_split_reproducible(self):
        tmp2  = tempfile.mkdtemp()
        paths = _run_pipeline(tmp2, seed=42)
        with open(paths["split_path"]) as f:
            split2 = json.load(f)
        self.assertEqual(self.split["test"],  split2["test"])
        self.assertEqual(self.split["train"], split2["train"])

    def test_different_seed_gives_different_split(self):
        tmp2  = tempfile.mkdtemp()
        paths = _run_pipeline(tmp2, seed=99)
        with open(paths["split_path"]) as f:
            split2 = json.load(f)
        self.assertNotEqual(self.split["test"], split2["test"])


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)