# Diagnostic and repair report (Sections 1–8)

Scope: **Green-only SUMO TLS programs** with **TraCI yellow + all-red** (per agreed plan). Embedded `<phase>` yellow/all-red rows in `.net.xml` are intentionally not used; B–E are satisfied via TraCI.

## 1. FIXES APPLIED

1. **1.6 / 3H** — Added `phase_program_mode`, `n_green_phases` to [`src/intersection.json`](src/intersection.json); kept `phases` as the string program mode (`"2"` / `"4"`). [`preprocessing/validate_intersection.py`](preprocessing/validate_intersection.py) computes and logs `n_green_phases`.
2. **1.7** — [`main.py`](main.py) overlays `min_green_s`, `max_green_s`, `amber_s` → `YELLOW_DURATION_S`, and `min_red_s` from intersection when the file exists; prints override when yellow changes. [`config.json`](config.json) `phase_timing.yellow_duration_s` set to `3.0` to match `amber_s`.
3. **1.4 `protected_some`** — [`preprocessing/BuildNetwork.py`](preprocessing/BuildNetwork.py) `build_phases` implements hybrid protected/permissive legs; validator cleans `protected_approaches`.
4. **3.E (TraCI)** — [`modelling/agent.py`](modelling/agent.py): `_run_all_red_clearance` after every full yellow before `setProgram` / `setPhase` (`_execute_green_switch`, `_execute_yellow_clearance`).
5. **4.2** — Explicit `_lockout_remaining` = `yellow_steps + min_green_steps` at SWITCH start; decremented each primitive SUMO step; forces KEEP while &gt; 0.
6. **4.3 / 4.6** — Agent advances **`decision_gap`** primitive steps per decision epoch (`_primitive_simulation_step` loop); yellow/all-red/min-green fast-forward remain **primitive** steps. [`main.py`](main.py) `TOTAL_UPDATES` uses `steps_per_ep / DECISION_GAP`.
7. **4.4 / 6.3–6.4** — [`ThroughputCompositeReward`](modelling/components/reward/throughput_composite.py): `switch_weight`, independent **tanh** normalisation per term when `normalise=True`, optional `queue_norm_cap`.
8. **4.1** — Documented SMDP attribution for min-green fast-forward in `Agent.step` docstring (no change to reward integration to avoid breaking throughput accounting).
9. **Build pipeline** — [`preprocessing/BuildNetwork.py`](preprocessing/BuildNetwork.py) `main()` and [`main.py`](main.py) `run_build_network` call `validate_and_fix_intersection` before building. TLS state sanitization via `_sanitize_tls_state_string` (3.G).
10. **Tooling / tests** — [`Testcases/Test_DiagnosticRepair.py`](Testcases/Test_DiagnosticRepair.py): intersection vs `build_phases`, `protected_some`, observation size, subslot mean, switch penalty, net/tl sanity, optional 60s SUMO smoke.

## 2. WARNINGS

- **`phases` vs count**: The key `phases` remains a **mode string**, not an integer phase count; use `n_green_phases` for counts.
- **Min-green reward (4.1)**: Rewards over min-green fast-forward still roll into the **previous** transition’s SMDP return; not a separate “no credit” MDP step.
- **Overshoot (4.5)**: Gaussian scaling of the reward past `max_green` remains in addition to the exponential hold penalty (not removed).

## 3. REMAINING ISSUES / HUMAN REVIEW

- **Tie-breaking / policy**: Baseline docs may still describe `min_green_steps` as `min_green_s / decision_gap`; Agent uses **primitive** steps for min-green and lockout — baseline `FixedTimePolicy` step counts may need retuning when `decision_gap &gt; 1`.
- **Webster / schedule**: If any code assumes one primitive step per agent step, re-verify with `decision_gap &gt; 1`.

## 4. VERIFICATION SUMMARY

| Check | Result | Notes |
|-------|--------|--------|
| 1.1 active_approaches count | **PASS** | Validator enforces 3 or 4. |
| 1.2 approaches completeness | **PASS** | Validator adds missing with medians + REVIEW log. |
| 1.3 lane keys / speed | **PASS** | Nested `lanes` + defaults in validator. |
| 1.4 left_turn_mode + protected_some | **PASS** | Enum + cleaning; `build_phases` implements `protected_some`. |
| 1.5 timing clamps | **PASS** | Validator clamps and logs. |
| 1.6 phases vs generated greens | **PASS** | `n_green_phases`; `phases` stays mode string. |
| 1.7 config vs intersection | **PASS** | Runtime overlay + `config.json` yellow aligned. |
| 2.1–2.4 net edges/lanes/speed/length | **PASS** | Covered by `Test_BuildNetwork` + `Test_DiagnosticRepair` when net present. |
| 2.5–2.6 connections / TLS id | **PASS** | Existing tests + net smoke. |
| 3.A green coverage | **PASS** | `build_phases` + TLL tests. |
| 3.B–D yellow in XML | **N/A** | TraCI yellow; duration matches `amber_s` via Agent. |
| 3.E all-red | **PASS** | TraCI `_run_all_red_clearance`. |
| 3.F left modes | **PASS** | Permissive / protected / protected_some in `build_phases`. |
| 3.G state strings | **PASS** | Sanitize invalid chars; tests assert valid charset + length. |
| 3.H phase count vs obs | **PASS** | `main` bumps `MAX_PHASE` from `n_green_phases`. |
| 4.1 min-green gate | **PARTIAL** | Enforced before policy; SMDP reward bundling (see Warnings). |
| 4.2 lockout | **PASS** | Primitive-step counter `yellow + min_green`. |
| 4.3 yellow / all-red order | **PASS** | Yellow → all-red → next green. |
| 4.4 switch penalty | **PASS** | `ThroughputCompositeReward.switch_weight`. |
| 4.5 max-green overshoot | **PASS** | Unchanged formula + optional Gaussian (see Warnings). |
| 4.6 decision_gap | **PASS** | `decision_gap` primitive steps per agent step. |
| 5.1 obs length | **PASS** | Project layout `max_lanes+6(+1)`; test asserts `size()` consistency. |
| 5.2 max_phase | **PASS** | Overlay from intersection. |
| 5.3 phase index | **PASS** | Raw `getPhase` in `QueueObservation`. |
| 5.4–5.5 cap then normalise | **PASS** | Existing `QueueObservation` logic. |
| 6.1–6.2 reward class / kwargs | **PASS** | `main` merge + config keys. |
| 6.3 composite terms | **PASS** | Departure-lane ΔV, mean queue, `pstdev`; + switch term. |
| 6.4 independent normalise | **PASS** | Tanh per term when `normalise`. |
| 6.5 after SUMO step | **PASS** | `compute` at decision boundary after advances. |
| 7.1 routes per approach | **PASS** | Flows use `edge_*_in` for all active legs (existing data). |
| 7.2 subslot expectation | **PASS** | Statistical test in `Test_DiagnosticRepair`. |
| 7.3 inactive edges | **PASS** | BuildRoute uses `_allowed_movements`. |
| 8.1 SUMO 60s | **PASS** | When SUMO available (`TestSumoSmoke`). |
| 8.2–8.5 full RL episode / SWITCH cases | **SKIPPED** | Not automated here; run manual or extend tests with TraCI scripting. |

---

*Generated as part of the diagnostic repair implementation (2026).*
