import os
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_system.schedule_controller import ScheduleController


class TestScheduleController(unittest.TestCase):
    def test_nearest_bucket_fallback(self):
        schedule = {
            "buckets": [
                {
                    "tls_id": "J_centre",
                    "phase": 0,
                    "bucket_start_s": 0.0,
                    "median_s": 10.0,
                    "std_s": 0.0,
                    "n": 1,
                },
                {
                    "tls_id": "J_centre",
                    "phase": 0,
                    "bucket_start_s": 1800.0,
                    "median_s": 20.0,
                    "std_s": 0.0,
                    "n": 1,
                },
            ]
        }
        ctl = ScheduleController(
            schedule=schedule,
            tls_id="J_centre",
            min_green_s=15.0,
            yellow_duration_s=3.0,
            step_length=1.0,
        )
        action = ctl.get_action(0, 900.0, 25.0)
        self.assertIn(action, (0, 1))


if __name__ == "__main__":
    unittest.main()
