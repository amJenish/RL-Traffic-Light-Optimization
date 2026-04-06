"""
Validate and optionally repair intersection.json (Sections 1.1–1.6).

Use from CLI or import validate_and_fix_intersection(cfg) before BuildNetwork.
"""

from __future__ import annotations

import copy
import json
import statistics
from typing import Any

def _import_build_phases():
    import os
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from preprocessing.BuildNetwork import build_phases

    return build_phases


def _lanes_dict(app: dict) -> dict[str, int]:
    if "lanes" in app and isinstance(app["lanes"], dict):
        return app["lanes"]
    return {
        k: int(app[k])
        for k in ("through", "right", "left")
        if k in app
    }


def _median_int(values: list[int], default: int = 1) -> int:
    if not values:
        return default
    return int(round(statistics.median(values)))


def validate_and_fix_intersection(
    cfg: dict[str, Any], *, fix: bool = True, log: print = print
) -> tuple[dict[str, Any], list[str]]:
    """
    Returns (possibly mutated cfg, warnings). If fix is False, only validates
    (raises ValueError on hard errors).
    """
    warnings: list[str] = []
    out = cfg if fix else copy.deepcopy(cfg)

    # Optional; BuildNetwork uses this as the .net.xml basename.
    _nm = out.get("intersection_name")
    if fix and (_nm is None or (isinstance(_nm, str) and not str(_nm).strip())):
        out["intersection_name"] = "My_Intersection"
        warnings.append("[name] intersection_name missing or empty; using default 'My_Intersection'")
        log(warnings[-1])

    active = out.get("active_approaches")
    if not isinstance(active, list) or not active:
        raise ValueError("active_approaches must be a non-empty list")
    n_active = len(active)
    if n_active not in (3, 4):
        raise ValueError(
            f"active_approaches must have 3 (T) or 4 (four-way) entries; got {n_active}"
        )

    approaches = out.setdefault("approaches", {})
    if not isinstance(approaches, dict):
        raise ValueError("approaches must be an object")

    # --- 1.2 missing approaches ---
    lane_counts_through: list[int] = []
    for d, app in approaches.items():
        if isinstance(app, dict):
            ld = _lanes_dict(app)
            lane_counts_through.append(int(ld.get("through", 1)))

    for d in active:
        if d not in approaches:
            med_t = max(1, _median_int(lane_counts_through, 1))
            med_r = _median_int(
                [
                    int(_lanes_dict(approaches[x]).get("right", 0))
                    for x in approaches
                    if isinstance(approaches.get(x), dict)
                ],
                0,
            )
            med_l = _median_int(
                [
                    int(_lanes_dict(approaches[x]).get("left", 0))
                    for x in approaches
                    if isinstance(approaches.get(x), dict)
                ],
                0,
            )
            msg = (
                f"[1.2] REVIEW: added missing approach {d!r} with median defaults "
                f"through={med_t} right={med_r} left={med_l}"
            )
            warnings.append(msg)
            log(msg)
            approaches[d] = {
                "lanes": {"through": med_t, "right": med_r, "left": med_l},
                "speed_kmh": 50,
            }

    # --- 1.3 per-approach lanes & speed ---
    for d in active:
        app = approaches[d]
        if not isinstance(app, dict):
            raise ValueError(f"approaches[{d!r}] must be an object")
        lanes = app.setdefault("lanes", {})
        if not isinstance(lanes, dict):
            raise ValueError(f"approaches[{d!r}].lanes must be an object")
        for mov, default in (("right", 0), ("left", 0)):
            if mov not in lanes:
                lanes[mov] = default
                warnings.append(f"[1.3] {d}.{mov} missing → filled {default}")
                log(warnings[-1])
        if "through" not in lanes:
            lanes["through"] = 1
            warnings.append("[1.3] through missing → 1")
            log(warnings[-1])
        for mov in ("through", "right", "left"):
            v = int(lanes[mov])
            lanes[mov] = max(0, v)
        if lanes["through"] < 1:
            orig = lanes["through"]
            lanes["through"] = 1
            warnings.append(f"[1.3] {d}.through {orig} < 1 → clamped to 1")
            log(warnings[-1])
        if "speed_kmh" not in app or not (float(app["speed_kmh"]) > 0):
            app["speed_kmh"] = 50.0
            warnings.append(f"[1.3] {d}.speed_kmh missing/invalid → 50")
            log(warnings[-1])

    # --- 1.4 left_turn_mode ---
    lt = str(out.get("left_turn_mode", "permissive")).strip().lower()
    if lt not in ("permissive", "protected", "protected_some"):
        warnings.append(f"[1.4] invalid left_turn_mode {lt!r} → permissive")
        log(warnings[-1])
        lt = "permissive"
        out["left_turn_mode"] = lt
    else:
        out["left_turn_mode"] = lt

    if lt == "protected_some":
        pa = out.get("protected_approaches")
        if not isinstance(pa, list):
            pa = []
            out["protected_approaches"] = pa
            warnings.append("[1.4] protected_some: protected_approaches missing → []")
            log(warnings[-1])
        valid_dirs = set(active)
        cleaned = []
        for x in pa:
            sx = str(x).upper() if isinstance(x, str) else str(x)
            if sx not in valid_dirs:
                warnings.append(
                    f"[1.4] protected_approaches removed invalid entry {x!r}"
                )
                log(warnings[-1])
                continue
            left_n = int(approaches[sx]["lanes"].get("left", 0))
            if left_n < 1:
                warnings.append(
                    f"[1.4] protected_approaches removed {sx!r} (left lanes < 1)"
                )
                log(warnings[-1])
                continue
            cleaned.append(sx)
        out["protected_approaches"] = cleaned

    # --- 1.5 timing clamps ---
    def clamp_warn(key: str, val: float, lo: float, hi: float | None) -> float:
        v = float(val)
        orig = v
        if hi is not None:
            v = min(max(v, lo), hi)
        else:
            v = max(v, lo)
        if v != orig:
            warnings.append(f"[1.5] {key} {orig} clamped → {v}")
            log(warnings[-1])
        return v

    mg = clamp_warn("min_green_s", out.get("min_green_s", 15), 7, None)
    out["min_green_s"] = int(mg)
    mx = float(out.get("max_green_s", 90))
    if mx <= out["min_green_s"]:
        mx = float(out["min_green_s"]) + 1.0
        warnings.append(f"[1.5] max_green_s adjusted to {mx} (> min_green_s)")
        log(warnings[-1])
    out["max_green_s"] = int(mx)

    amb = clamp_warn("amber_s", out.get("amber_s", 3), 3, 5)
    out["amber_s"] = int(amb)

    mr = clamp_warn("min_red_s", out.get("min_red_s", 1), 1, None)
    out["min_red_s"] = int(mr)

    cy = clamp_warn("cycle_s", out.get("cycle_s", 120), 1e-3, None)
    out["cycle_s"] = int(cy)

    el = clamp_warn("edge_length_m", out.get("edge_length_m", 200), 1e-3, None)
    out["edge_length_m"] = float(el)

    # --- 1.6 phase program mode + n_green_phases (do not replace phases int) ---
    mode = str(out.get("phases", "2"))
    out.setdefault("phase_program_mode", mode)
    build_phases = _import_build_phases()
    prot = out.get("protected_approaches") if lt == "protected_some" else None
    greens = build_phases(
        list(active),
        mode,
        out["amber_s"],
        out["min_green_s"],
        out["cycle_s"],
        approaches=out.get("approaches", {}),
        left_turn_mode=lt,
        protected_approaches=prot if isinstance(prot, list) else None,
    )
    out["n_green_phases"] = len(greens)
    log(f"[1.6] n_green_phases = {out['n_green_phases']} (phase_program_mode={mode!r})")

    return out, warnings


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate intersection.json")
    p.add_argument("--config", required=True)
    p.add_argument("--write", action="store_true", help="Write fixed JSON back")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    validate_and_fix_intersection(cfg, fix=True, log=print)
    if args.write:
        with open(args.config, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"Wrote {args.config}")


if __name__ == "__main__":
    main()
