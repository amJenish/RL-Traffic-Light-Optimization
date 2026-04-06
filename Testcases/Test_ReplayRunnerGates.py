import os
import sys
import unittest
from unittest.mock import patch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eval_system.replay_runner as rr


class _Phase:
    def __init__(self, state: str):
        self.state = state


class _Logic:
    def __init__(self):
        self.programID = "0"
        self.phases = [_Phase("Gr"), _Phase("rG")]


class _Simulation:
    def __init__(self):
        self.t = 0.0

    def getTime(self):
        return self.t


class _TrafficLight:
    def __init__(self, sim: _Simulation):
        self._sim = sim
        self._logic = _Logic()
        self._phase = 0
        self._live_state = self._logic.phases[self._phase].state
        self.yellow_starts: list[float] = []

    def getIDList(self):
        return ["J_centre"]

    def getAllProgramLogics(self, tls_id):
        _ = tls_id
        return [self._logic]

    def setProgram(self, tls_id, pid):
        _ = tls_id, pid

    def setPhase(self, tls_id, phase):
        _ = tls_id
        self._phase = int(phase)
        self._live_state = self._logic.phases[self._phase].state

    def getPhase(self, tls_id):
        _ = tls_id
        return self._phase

    def getRedYellowGreenState(self, tls_id):
        _ = tls_id
        return self._live_state

    def setRedYellowGreenState(self, tls_id, state):
        _ = tls_id
        self._live_state = state
        if "y" in state:
            self.yellow_starts.append(self._sim.getTime())

    def getControlledLanes(self, tls_id):
        _ = tls_id
        return ["lane_a"]

    def getControlledLinks(self, tls_id):
        _ = tls_id
        return []


class _Lane:
    def getLastStepVehicleIDs(self, lane):
        _ = lane
        return []

    def getWaitingTime(self, lane):
        _ = lane
        return 0.0


class _Edge:
    def getIDList(self):
        return []

    def getFromID(self, eid):
        _ = eid
        return ""

    def getLaneNumber(self, eid):
        _ = eid
        return 0


class _FakeTraci:
    class exceptions:
        class TraCIException(Exception):
            pass

    def __init__(self):
        self.simulation = _Simulation()
        self.trafficlight = _TrafficLight(self.simulation)
        self.lane = _Lane()
        self.edge = _Edge()


class _FakeEnv:
    def __init__(self, **kwargs):
        self._end = float(kwargs["end"])
        self._traci = _FakeTraci()

    @property
    def traci(self):
        return self._traci

    def start(self, route_file: str):
        _ = route_file

    def step(self, n_steps=1):
        self._traci.simulation.t += float(n_steps)

    def is_done(self):
        return self._traci.simulation.getTime() >= self._end

    def close(self):
        return None


class _AlwaysSwitchController:
    def get_action(self, current_green_phase, sim_time, time_in_phase):
        _ = current_green_phase, sim_time, time_in_phase
        return 1


class TestReplayRunnerGates(unittest.TestCase):
    def test_min_green_and_lockout_are_enforced(self):
        with patch.object(rr, "SumoEnvironment", _FakeEnv):
            runner = rr.ReplayRunner(
                net_file="dummy.net.xml",
                step_length=1.0,
                decision_gap=1,
                begin=0,
                end=40,
                sumo_home="dummy",
                gui=False,
            )
            result = runner.run(
                flow_file="dummy.rou.xml",
                controller=_AlwaysSwitchController(),
                min_green_s=5.0,
                yellow_duration_s=2.0,
                min_red_s=1.0,
                tls_id="J_centre",
                label="DQN",
            )

            starts = runner._env.traci.trafficlight.yellow_starts
            self.assertTrue(starts)
            self.assertGreaterEqual(starts[0], 5.0)  # min green gate
            for a, b in zip(starts, starts[1:]):
                self.assertGreaterEqual(
                    b - a, 7.0
                )  # lockout = yellow + min_green
            self.assertGreaterEqual(result.phase_switch_count, 1)


if __name__ == "__main__":
    unittest.main()
