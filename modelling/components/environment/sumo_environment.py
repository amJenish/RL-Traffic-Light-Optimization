"""SUMO/TraCI environment wrapper. Manages start -> step -> done -> close."""

import os
import sys
import uuid
from typing import Any

from .base import BaseEnvironment


class SumoEnvironment(BaseEnvironment):
    """Wraps a SUMO simulation via TraCI."""

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
        self._traci_view   = None
        self._traci_module = None
        self._label        = None
        self._running      = False

        self._validate_sumo_home()
        self._import_traci()

    def start(self, route_file: str) -> None:
        """Launch SUMO for one episode using the given route file."""
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

        self._label = f"sumo_{uuid.uuid4().hex[:8]}"
        self._traci_module.start(cmd, label=self._label)
        self._traci = self._traci_module.getConnection(self._label)
        self._traci_view = _TraCIConnectionView(self._traci, self._traci_module)
        self._running = True

    def step(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            if not self.is_done():
                self._traci.simulationStep()

    def is_done(self) -> bool:
        if not self._running:
            return True
        return self._traci.simulation.getTime() >= self._end

    def close(self) -> None:
        if self._running and self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                # Best-effort fallback: close default module connection too.
                try:
                    self._traci_module.close()
                except Exception:
                    pass
        self._traci = None
        self._traci_view = None
        self._label = None
        self._running = False

    def get_tls_ids(self) -> list[str]:
        return list(self._traci.trafficlight.getIDList())

    def get_sim_time(self) -> float:
        return self._traci.simulation.getTime()

    @property
    def traci(self) -> Any:
        return self._traci_view

    def step_decision(self) -> None:
        """Advance by one full decision interval (legacy, used before adaptive timing)."""
        self.step(self._decision_gap)

    def _import_traci(self) -> None:
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


class _TraCIConnectionView:
    """Connection facade that preserves module-level helpers like `.exceptions`."""

    def __init__(self, conn: Any, module: Any) -> None:
        self._conn = conn
        self.exceptions = module.exceptions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)
