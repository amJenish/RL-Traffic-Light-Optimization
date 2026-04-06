import os
import sys
import tempfile
import unittest

import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modelling.webster_schedule_builder import WebsterScheduleBuilder


class TestWebsterScheduleBuilder(unittest.TestCase):
    def test_three_day_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            icfg = {
                "intersection_name": "X",
                "active_approaches": ["N", "S", "E", "W"],
                "approaches": {
                    "N": {"lanes": {"through": 2, "right": 1, "left": 1}},
                    "S": {"lanes": {"through": 2, "right": 1, "left": 1}},
                    "E": {"lanes": {"through": 2, "right": 1, "left": 1}},
                    "W": {"lanes": {"through": 2, "right": 1, "left": 1}},
                },
                "phases": "4",
                "n_green_phases": 2,
                "left_turn_mode": "permissive",
                "min_green_s": 15,
                "max_green_s": 90,
                "amber_s": 3,
                "cycle_s": 120,
            }
            rows = [
                {
                    "sim_day": 0,
                    "start_time": "08:00:00",
                    "N_through": 200,
                    "N_right": 80,
                    "N_left": 40,
                    "S_through": 210,
                    "S_right": 70,
                    "S_left": 35,
                    "E_through": 150,
                    "E_right": 60,
                    "E_left": 30,
                    "W_through": 160,
                    "W_right": 55,
                    "W_left": 25,
                },
                {
                    "sim_day": 0,
                    "start_time": "08:15:00",
                    "N_through": 220,
                    "N_right": 85,
                    "N_left": 45,
                    "S_through": 225,
                    "S_right": 75,
                    "S_left": 38,
                    "E_through": 145,
                    "E_right": 58,
                    "E_left": 29,
                    "W_through": 155,
                    "W_right": 57,
                    "W_left": 28,
                },
            ]
            paths = []
            for d in range(3):
                df = pd.DataFrame(rows)
                df["sim_day"] = d
                path = os.path.join(td, f"day_{d:02d}.csv")
                df.to_csv(path, index=False)
                paths.append(path)

            b = WebsterScheduleBuilder(icfg)
            sched = b.build_from_day_csvs(paths, "J_centre", slot_minutes=15)
            buckets = sched["buckets"]
            self.assertEqual(len(buckets), 4)  # 2 slots * 2 phases

            for row in buckets:
                self.assertGreaterEqual(row["median_s"], icfg["min_green_s"])
                self.assertLessEqual(row["median_s"], icfg["max_green_s"])
                self.assertEqual(row["n"], 3)

            phases = b._green_phases()
            n = len(phases)
            lost = n * (icfg["amber_s"] + 1.0)
            c_min = n * icfg["min_green_s"] + lost
            c_max = n * icfg["max_green_s"] + lost
            for row in rows:
                greens = b._compute_slot_greens(
                    pd.Series(row),
                    phases,
                    15,
                    icfg["approaches"],
                    icfg["amber_s"],
                    icfg["min_green_s"],
                    icfg["max_green_s"],
                )
                cycle = sum(greens) + lost
                self.assertGreaterEqual(cycle, c_min - 1e-6)
                self.assertLessEqual(cycle, c_max + 1e-6)
                self.assertLessEqual(sum(greens), (cycle - lost) + 0.01)


if __name__ == "__main__":
    unittest.main()
