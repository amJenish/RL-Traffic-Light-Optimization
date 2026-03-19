"""
modeling/components/environment/sumo_environment.py
-----------------------------------------------------
SUMO/TraCI implementation of BaseEnvironment.

Manages the full lifecycle of a SUMO simulation episode:
start → step → done → close.

Reads intersection and network config from split.json so it stays
consistent with what BuildRoute and BuildNetwork produced.
"""

import os
import subprocess
import sys
from typing import Any

from .base import BaseEnvironment


class SumoEnvironment(BaseEnvironment):
    """
    Wraps a SUMO simulation via TraCI.

    Args:
        net_file:        Path to the .net.xml network file.
        step_length:     Simulation step size in seconds (default 1).
        decision_gap:    How many simulation steps between agent decisions.
        gui:             If True, launches sumo-gui instead of sumo.
        sumo_home:       Path to SUMO installation. If None, uses SUMO_HOME
                         environment variable.
    """

    def __init__(
        self,
        net_file:     str,
        step_length:  float = 1.0,
        decision_gap: int   = 10,
        gui:          bool  = False,
        sumo_home:    str | None = None,
        begin:        int   = 27000,
        end:          int   = 64800,
    ):
        self._net_file     = net_file
        self._step_length  = step_length
        self._decision_gap = decision_gap
        self._gui          = gui
        self._sumo_home    = sumo_home or os.environ.get("SUMO_HOME", "")
        self._begin        = begin
        self._end          = end
        self._traci        = None
        self._traci_module = None
        self._running      = False

        self._validate_sumo_home()
        self._import_traci()  # import eagerly so TraCI is always available


    # ABSTRACT METHOD IMPLEMENTATIONS

    def start(self, route_file: str) -> None:
        """Start SUMO for a new episode using the given route file."""
        if self._running:
            self.close()

        binary = "sumo-gui" if self._gui else "sumo"
        binary_path = os.path.join(self._sumo_home, "bin", binary)

        cmd = [
            binary_path,
            "--net-file",          self._net_file,
            "--route-files",       route_file,
            "--step-length",       str(self._step_length),
            "--begin",             str(self._begin),
            "--end",               str(self._end),
            "--no-warnings",       "true",
            "--no-step-log",       "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport",  "-1",
        ]

        self._traci_module.start(cmd)
        self._running = True

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n_steps × step_length seconds."""
        for _ in range(n_steps):
            if not self.is_done():
                self._traci_module.simulationStep()

    def is_done(self) -> bool:
        if not self._running:
            return True
        return self._traci_module.simulation.getTime() >= self._end

    def close(self) -> None:
        """Shut down TraCI and SUMO."""
        if self._running and self._traci_module:
            try:
                self._traci_module.close()
            except Exception:
                pass
        self._running = False

    def get_tls_ids(self) -> list[str]:
        """Return all traffic light IDs in the loaded network."""
        return list(self._traci_module.trafficlight.getIDList())

    def get_sim_time(self) -> float:
        """Return current simulation time in seconds."""
        return self._traci_module.simulation.getTime()

    @property
    def traci(self) -> Any:
        """Raw TraCI module for observation builders and reward functions."""
        return self._traci_module


    # DECISION GAP

    def step_decision(self) -> None:
        """
        Advance by one full decision interval (decision_gap steps).
        Called by the Agent between each action.
        """
        self.step(self._decision_gap)

    # PRIVATE HELPERS

    def _import_traci(self) -> None:
        """Import TraCI — deferred so the module loads without SUMO installed."""
        if self._traci_module is not None:
            return
        tools = os.path.join(self._sumo_home, "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        try:
            import traci as _traci
            self._traci_module = _traci
        except ImportError as e:
            raise ImportError(
                f"Could not import TraCI. "
                f"Make sure SUMO is installed and SUMO_HOME is set.\n"
                f"SUMO_HOME={self._sumo_home}\n"
                f"Original error: {e}"
            )

    def _validate_sumo_home(self) -> None:
        if not self._sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Either pass sumo_home= to SumoEnvironment "
                "or set the SUMO_HOME environment variable to SUMO installation path.\n"
                "Example: C:\\Program Files (x86)\\Eclipse\\Sumo"
            )