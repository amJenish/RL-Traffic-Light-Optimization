from __future__ import annotations

from typing import Any

from modelling.components.environment.sumo_environment import SumoEnvironment

from eval_system.kpi_collector import KPICollector, KPIResult
from eval_system.schedule_controller import ScheduleController

_STATIC_PROGRAM_ID = "0"


def _static_program_logic(traci: Any, tls_id: str) -> Any:
    for lg in traci.trafficlight.getAllProgramLogics(tls_id):
        pid = getattr(lg, "programID", None) or getattr(lg, "subID", None)
        if str(pid) == _STATIC_PROGRAM_ID:
            return lg
    return traci.trafficlight.getAllProgramLogics(tls_id)[0]


def _is_yellow_phase(phase_state: str) -> bool:
    return (
        "y" in phase_state
        and "G" not in phase_state
        and "g" not in phase_state
    )


def _yellow_clearance_state_between(prev_state: str, next_state: str) -> str:
    out: list[str] = []
    for p, n in zip(prev_state, next_state):
        if p in "Gg" and n == "r":
            out.append("y")
        else:
            out.append("r")
    return "".join(out)


class ReplayRunner:
    def __init__(
        self,
        net_file: str,
        step_length: float,
        decision_gap: int,
        begin: int,
        end: int,
        sumo_home: str,
        gui: bool = False,
    ) -> None:
        self._net_file = net_file
        self._step_length = float(step_length)
        self._decision_gap = max(1, int(decision_gap))
        self._begin = int(begin)
        self._end = int(end)
        self._sumo_home = sumo_home
        self._gui = gui
        self._env = SumoEnvironment(
            net_file=net_file,
            step_length=self._step_length,
            decision_gap=self._decision_gap,
            gui=gui,
            sumo_home=sumo_home,
            begin=self._begin,
            end=self._end,
        )

    def _step_kpi(
        self,
        traci: Any,
        kpi: KPICollector,
        *,
        accumulate: bool,
    ) -> None:
        if self._env.is_done():
            return
        self._env.step(1)
        kpi.on_step(traci, accumulate=accumulate)

    def run(
        self,
        flow_file: str,
        controller: ScheduleController,
        min_green_s: float,
        yellow_duration_s: float,
        min_red_s: float,
        tls_id: str,
        label: str,
    ) -> KPIResult:
        yellow_steps = max(1, round(float(yellow_duration_s) / self._step_length))
        min_red_steps = max(1, round(float(min_red_s) / self._step_length))
        min_green_s = float(min_green_s)

        kpi = KPICollector(tls_id)
        phase_switch_count = 0
        time_in_phase = 0.0
        lockout_remaining = 0.0
        yellow_left = 0
        all_red_left = 0
        next_green_phase: int | None = None
        switch_len = 0

        try:
            self._env.start(flow_file)
            traci = self._env.traci
            kpi.reset()

            warmup_steps = max(5, round(25 / self._step_length))
            for _ in range(warmup_steps):
                if self._env.is_done():
                    break
                self._step_kpi(traci, kpi, accumulate=False)

            for tid in list(traci.trafficlight.getIDList()):
                traci.trafficlight.setProgram(tid, _STATIC_PROGRAM_ID)
                traci.trafficlight.setPhase(tid, 0)

            step_idx = 0

            while not self._env.is_done():
                sim_time = float(traci.simulation.getTime())

                if yellow_left > 0:
                    self._step_kpi(traci, kpi, accumulate=True)
                    step_idx += 1
                    time_in_phase += self._step_length
                    lockout_remaining = max(
                        0.0, lockout_remaining - self._step_length
                    )
                    yellow_left -= 1
                    if yellow_left == 0:
                        logic = _static_program_logic(traci, tls_id)
                        slen = switch_len or len(logic.phases[0].state)
                        traci.trafficlight.setRedYellowGreenState(
                            tls_id, "r" * int(slen)
                        )
                        all_red_left = min_red_steps
                    continue

                if all_red_left > 0:
                    self._step_kpi(traci, kpi, accumulate=True)
                    step_idx += 1
                    time_in_phase += self._step_length
                    lockout_remaining = max(
                        0.0, lockout_remaining - self._step_length
                    )
                    all_red_left -= 1
                    if all_red_left == 0 and next_green_phase is not None:
                        traci.trafficlight.setProgram(
                            tls_id, _STATIC_PROGRAM_ID
                        )
                        traci.trafficlight.setPhase(tls_id, int(next_green_phase))
                        next_green_phase = None
                    continue

                self._step_kpi(traci, kpi, accumulate=True)
                step_idx += 1
                time_in_phase += self._step_length
                lockout_remaining = max(
                    0.0, lockout_remaining - self._step_length
                )

                try:
                    live = traci.trafficlight.getRedYellowGreenState(tls_id)
                except Exception:
                    live = ""
                if _is_yellow_phase(live):
                    continue

                if step_idx % self._decision_gap != 0:
                    continue

                logic = _static_program_logic(traci, tls_id)
                n_phases = len(logic.phases)
                cur = int(traci.trafficlight.getPhase(tls_id))
                if cur >= n_phases:
                    traci.trafficlight.setProgram(tls_id, _STATIC_PROGRAM_ID)
                    cur = min(int(traci.trafficlight.getPhase(tls_id)), n_phases - 1)
                try:
                    live2 = traci.trafficlight.getRedYellowGreenState(tls_id)
                except Exception:
                    live2 = logic.phases[cur].state
                if _is_yellow_phase(live2):
                    continue

                action = controller.get_action(cur, sim_time, time_in_phase)
                if time_in_phase < min_green_s:
                    action = 0
                if lockout_remaining > 0:
                    action = 0

                if action == 1:
                    prev_st = logic.phases[cur].state
                    nxt = (cur + 1) % n_phases
                    next_st = logic.phases[nxt].state
                    y_st = _yellow_clearance_state_between(prev_st, next_st)
                    switch_len = len(y_st)
                    traci.trafficlight.setRedYellowGreenState(tls_id, y_st)
                    yellow_left = yellow_steps
                    next_green_phase = nxt
                    lockout_remaining = float(yellow_duration_s) + min_green_s
                    time_in_phase = 0.0
                    phase_switch_count += 1

            elapsed_s = float(self._end - self._begin)
            return kpi.collect(
                elapsed_s,
                label=label,
                phase_switch_count=phase_switch_count,
            )
        finally:
            self._env.close()
