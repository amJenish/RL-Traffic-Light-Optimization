"""Diagnostic / repair plan checks (Sections 1, 5, 6, 7.2, 8 smoke)."""

import json
import math
import os
import random
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PATH_INTERSECTION = os.path.join(PROJECT_ROOT, "src", "intersection.json")
PATH_BUILDNETWORK = os.path.join(PROJECT_ROOT, "preprocessing", "BuildNetwork.py")

import importlib.util


def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bn = _load_module("BuildNetwork", PATH_BUILDNETWORK)


class TestValidateIntersection(unittest.TestCase):
    def test_n_green_phases_matches_build(self):
        from preprocessing.validate_intersection import validate_and_fix_intersection

        with open(PATH_INTERSECTION, encoding="utf-8") as f:
            cfg = json.load(f)
        fixed, _w = validate_and_fix_intersection(cfg, fix=True, log=lambda *_: None)
        n = fixed["n_green_phases"]
        mode = str(fixed.get("phases", "2"))
        lt = str(fixed.get("left_turn_mode", "permissive"))
        prot = fixed.get("protected_approaches")
        if not isinstance(prot, list):
            prot = None
        greens = bn.build_phases(
            fixed["active_approaches"],
            mode,
            int(fixed.get("amber_s", 3)),
            int(fixed.get("min_green_s", 15)),
            int(fixed.get("cycle_s", 120)),
            approaches=fixed.get("approaches", {}),
            left_turn_mode=lt,
            protected_approaches=prot,
        )
        self.assertEqual(n, len(greens))


class TestProtectedSomePhases(unittest.TestCase):
    def test_only_listed_gets_left_bar(self):
        approaches = {
            "N": {"lanes": {"through": 2, "right": 1, "left": 1}, "speed_kmh": 50},
            "S": {"lanes": {"through": 2, "right": 1, "left": 1}, "speed_kmh": 50},
            "E": {"lanes": {"through": 2, "right": 1, "left": 1}, "speed_kmh": 50},
            "W": {"lanes": {"through": 2, "right": 1, "left": 1}, "speed_kmh": 50},
        }
        phases = bn.build_phases(
            ["N", "S", "E", "W"],
            "4",
            3,
            15,
            120,
            approaches=approaches,
            left_turn_mode="protected_some",
            protected_approaches=["N"],
        )
        names = [p["name"] for p in phases]
        self.assertIn("NS_left_prot", names)
        self.assertIn("NS_service", names)
        # N is protected-left only in NS_left_prot; NS_service should not repeat N left
        svc = next(p for p in phases if p["name"] == "NS_service")
        self.assertNotIn(("N", "left"), svc["green_groups"])
        self.assertIn(("S", "left"), svc["green_groups"])


class TestQueueObservationSize(unittest.TestCase):
    def test_build_matches_size(self):
        from modelling.components.observation.queue_observation import QueueObservation

        obs = QueueObservation(
            max_lanes=16,
            max_phase=7,
            max_phase_time=120.0,
            max_vehicles=20,
            max_green_s=90.0,
        )
        self.assertEqual(obs.size(), 16 + 6 + 1)


class TestSubslotRenormalisation(unittest.TestCase):
    def test_expected_mean_matches_base_vph(self):
        """Section 7.2: mean vehsPerHour across sub-slots equals base_vph (mean w = 1)."""
        import importlib.util
        import os

        br_path = os.path.join(PROJECT_ROOT, "preprocessing", "BuildRoute.py")
        spec = importlib.util.spec_from_file_location("BuildRoute", br_path)
        br = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(br)

        k = 5
        spread = 0.85
        base = 100.0
        n_trials = 2000
        means = []
        for _ in range(n_trials):
            rng = random.Random()
            w = br._bounded_weights(k, spread, rng)
            vph_list = [base * wi for wi in w]
            means.append(sum(vph_list) / k)
        avg = sum(means) / len(means)
        self.assertAlmostEqual(avg, base, delta=0.01)


class TestThroughputComposite(unittest.TestCase):
    def test_switch_penalty_subtracted(self):
        from modelling.components.reward.throughput_composite import (
            ThroughputCompositeReward,
        )

        r = ThroughputCompositeReward(
            normalise=False, switch_weight=2.0, scale=1.0, gamma=0.0, alpha=0.0, beta=0.0
        )

        class _Traci:
            class trafficlight:
                @staticmethod
                def getControlledLanes(_tls):
                    return []

            class lane:
                @staticmethod
                def getLastStepHaltingNumber(_lane):
                    return 0

        v = r.compute(_Traci(), "J1", switched=True)
        self.assertEqual(v, -2.0)


class TestNetXmlIfPresent(unittest.TestCase):
    def test_tls_and_in_edges(self):
        net = os.path.join(
            PROJECT_ROOT, "src", "data", "sumo", "network", "McCowan_Finch.net.xml"
        )
        if not os.path.isfile(net):
            self.skipTest("McCowan_Finch.net.xml not built")
        import xml.etree.ElementTree as ET

        tree = ET.parse(net)
        root = tree.getroot()
        tls = root.findall(".//tlLogic")
        self.assertTrue(tls, "tlLogic missing")
        self.assertEqual(tls[0].get("id"), "J_centre")
        edges = {e.get("id") for e in root.findall("edge") if not str(e.get("id", "")).startswith(":")}
        for d in ("N", "S", "E", "W"):
            self.assertIn(f"edge_{d}_in", edges, f"missing inbound {d}")


def _sumo_bin() -> str | None:
    home = os.environ.get("SUMO_HOME", "").strip()
    if home:
        for name in ("sumo", "sumo.exe"):
            p = os.path.join(home, "bin", name)
            if os.path.isfile(p):
                return p
    import shutil

    return shutil.which("sumo") or shutil.which("sumo.exe")


@unittest.skipUnless(_sumo_bin(), "SUMO not on PATH / SUMO_HOME")
class TestSumoSmoke(unittest.TestCase):
    def test_one_minute_sumo(self):
        import subprocess

        net = os.path.join(
            PROJECT_ROOT, "src", "data", "sumo", "network", "McCowan_Finch.net.xml"
        )
        flow = os.path.join(
            PROJECT_ROOT, "src", "data", "sumo", "flows", "flows_day_00.rou.xml"
        )
        if not (os.path.isfile(net) and os.path.isfile(flow)):
            self.skipTest("net or flow missing")
        binary = _sumo_bin()
        cmd = [
            binary,
            "-n",
            net,
            "-r",
            flow,
            "--end",
            "60",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, msg=r.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
