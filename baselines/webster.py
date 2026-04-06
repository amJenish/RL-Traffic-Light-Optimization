"""
WebsterPolicy - classical Webster (1958) fixed-time + gap-out hybrid.
Reads directly from QueueObservation vectors; requires the intersection
config dict (intersection.json) to build lane to phase mappings at init.
TLS programs are green-only rows; clearance uses TraCI (see Agent).
"""

import numpy as np
from modelling.components.policy.base import BasePolicy
from modelling.components.replay_buffer.base import BaseReplayBuffer

_TURN_TARGETS = {
    "N": {"right": "W", "through": "S", "left": "E"},
    "S": {"right": "E", "through": "N", "left": "W"},
    "E": {"right": "N", "through": "W", "left": "S"},
    "W": {"right": "S", "through": "E", "left": "N"},
}


def _build_lane_phase_map(cfg: dict) -> list[list[int]]:
    active    = cfg["active_approaches"]
    approaches = cfg.get("approaches", {})
    mode      = str(cfg.get("phases", "2"))
    lt_mode   = (cfg.get("left_turn_mode") or "permissive").strip().lower()
    if lt_mode not in ("permissive", "protected"):
        lt_mode = "permissive"
    use_perm  = mode == "2" or lt_mode == "permissive"
    lane_index = 0
    approach_lane_indices: dict[str, dict[str, list[int]]] = {}
    for d in ("N", "E", "S", "W"):
        if d not in active:
            continue
        lanes_cfg = approaches.get(d, {}).get("lanes", {"through": 1})
        approach_lane_indices[d] = {"right": [], "through": [], "left": []}
        for movement in ["right", "through", "left"]:
            count = lanes_cfg.get(movement, 0)
            target = _TURN_TARGETS[d][movement]
            if target not in active:
                lane_index += count
                continue
            for _ in range(count):
                approach_lane_indices[d][movement].append(lane_index)
                lane_index += 1
    ns_dirs = [d for d in ("N", "S") if d in active]
    ew_dirs = [d for d in ("E", "W") if d in active]

    def _has_left(dirs):
        return any(
            approaches.get(d, {}).get("lanes", {}).get("left", 0) > 0
            for d in dirs
        )

    def _collect(dirs, movements):
        idx = []
        for d in dirs:
            for m in movements:
                idx.extend(approach_lane_indices.get(d, {}).get(m, []))
        return idx

    groups: list[list[int]] = []
    if use_perm:
        if ns_dirs:
            groups.append(_collect(ns_dirs, ["right", "through", "left"]))
        if ew_dirs:
            groups.append(_collect(ew_dirs, ["right", "through", "left"]))
    else:
        if ns_dirs:
            if _has_left(ns_dirs):
                groups.append(_collect(ns_dirs, ["left"]))
            groups.append(_collect(ns_dirs, ["right", "through"]))
        if ew_dirs:
            if _has_left(ew_dirs):
                groups.append(_collect(ew_dirs, ["left"]))
            groups.append(_collect(ew_dirs, ["right", "through"]))
    return groups


class WebsterPolicy(BasePolicy):
    def __init__(
        self,
        cfg:             dict,
        max_lanes:       int   = 16,
        max_phase:       int   = 3,
        max_phase_time:  float = 120.0,
        critical_gap:    float = 3.5,
        step_duration:   float = 1.0,
    ):
        self._max_lanes      = max_lanes
        self._max_phase      = max_phase
        self._max_phase_time = max_phase_time
        self._critical_gap   = critical_gap
        self._step_duration  = step_duration
        self._eval_mode      = False
        self._min_green = float(cfg.get("min_green_s", 15))
        self._max_green = float(cfg.get("max_green_s", 90))
        self._amber_s   = float(cfg.get("amber_s", 3))
        self._lost_time = self._amber_s
        self._phase_lane_groups = _build_lane_phase_map(cfg)
        self._n_green_phases    = len(self._phase_lane_groups)

    def _compute_green_times(self, phase_pressures: np.ndarray) -> np.ndarray:
        y   = np.clip(phase_pressures, 0.0, 1.0)
        Y   = y.sum()
        n   = self._n_green_phases
        L   = n * self._lost_time
        if Y < 1e-6:
            return np.full(n, self._min_green)
        if Y >= 1.0:
            C = self._max_green * n
        else:
            C = (1.5 * L + 5.0) / (1.0 - Y)
        C = np.clip(C, self._min_green * n, self._max_green * n)
        effective_green = C - L
        greens = effective_green * (y / Y)
        return np.clip(greens, self._min_green, self._max_green)

    def _sumo_phase_to_green_index(self, sumo_phase: int) -> int | None:
        if sumo_phase < 0 or sumo_phase >= self._n_green_phases:
            return None
        return int(sumo_phase)

    def select_action(self, obs: np.ndarray, tls_id: str = "default") -> int:
        halting     = obs[:self._max_lanes]
        sumo_phase  = round(obs[self._max_lanes] * self._max_phase)
        time_in_phase = obs[self._max_lanes + 1] * self._max_phase_time
        green_idx = self._sumo_phase_to_green_index(sumo_phase)
        if green_idx is None:
            return 0
        pressures = np.array([
            halting[idxs].sum() / max(len(idxs), 1)
            for idxs in self._phase_lane_groups
        ])
        green_times = self._compute_green_times(pressures)
        target_green = min(
            float(green_times[green_idx]), float(self._max_phase_time)
        )
        if time_in_phase < self._min_green:
            return 0
        current_pressure = pressures[green_idx]
        headway = (self._step_duration / current_pressure) \
                  if current_pressure > 1e-6 else 9999.0
        if headway > self._critical_gap or time_in_phase >= target_green:
            return 1
        return 0

    def update(self, replay_buffer) -> float | None:
        return None

    def save(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "min_green":    self._min_green,
                "max_green":    self._max_green,
                "amber_s":      self._amber_s,
                "critical_gap": self._critical_gap,
                "n_green_phases": self._n_green_phases,
            }, f, indent=2)

    def load(self, path: str) -> None:
        import json
        with open(path) as f:
            ck = json.load(f)
        self._min_green    = ck["min_green"]
        self._max_green    = ck["max_green"]
        self._amber_s      = ck["amber_s"]
        self._critical_gap = ck["critical_gap"]

    def set_eval_mode(self)       -> None: self._eval_mode = True
    def set_train_mode(self)      -> None: self._eval_mode = False
    def reset_phase_tracking(self)-> None: pass

    @property
    def epsilon(self) -> float:
        return 0.0

    @property
    def device(self) -> str:
        return "cpu"
