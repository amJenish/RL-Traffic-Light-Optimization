# Project Handoff — RL Traffic Light Optimization
### Capstone Project Status Document

This document is written for a Cursor agent picking up this project on a new machine.
It describes everything that has been built, the current state of the codebase,
and the exact next steps remaining.

---

## What this project is

A reinforcement learning system that trains a DQN agent to control a traffic signal
at the **McCowan Rd / Finch Ave E** intersection in Toronto.
The agent runs inside the SUMO traffic simulator and learns to minimize
the number of halting vehicles by deciding at each step whether to hold
the current signal phase or advance to the next one.

The project also has two rule-based **baseline controllers** (Fixed-Time and Actuated)
that run in the same SUMO environment on the same test days,
so the DQN results can be compared against them for the capstone report.

---

## Current repository structure

```
RL-Traffic-Light-Optimization/
│
├── main.py                          ← Full pipeline entry point (preprocess → train → evaluate)
├── config.json                      ← All hyperparameters
├── requirements.txt                 ← Python dependencies
├── PIPELINE_OVERVIEW.md             ← Architecture diagram + file-by-file reference
├── HANDOFF.md                       ← This file
│
├── src/
│   ├── intersection.json            ← Intersection geometry + phase config (McCowan/Finch)
│   ├── columns.json                 ← Maps CSV column names to standard names
│   └── data/
│       ├── synthetic_toronto_data.csv
│       ├── processed/
│       │   ├── split.json           ← Train/test day split (train: 50 days, test: 6 days)
│       │   └── days/day_NN.csv      ← Per-day traffic CSVs (56 days total)
│       ├── sumo/
│       │   ├── network/             ← Built SUMO network (McCowan_Finch.net.xml + support files)
│       │   └── flows/               ← SUMO demand files (flows_day_00.rou.xml … flows_day_55.rou.xml)
│       └── models/
│           ├── final_model.pt       ← Trained DQN weights (3 epochs, 150 episodes)
│           ├── checkpoint_ep*.pt    ← Checkpoints every 10 episodes
│           ├── train_log.json       ← Per-episode reward/loss/epsilon for all 150 train episodes
│           └── test_log.json        ← DQN results on 6 held-out test days  ← ALREADY DONE
│
├── preprocessing/
│   ├── BuildRoute.py                ← Converts CSV → per-day CSVs + SUMO flow files + split.json
│   ├── BuildNetwork.py              ← Builds SUMO .net.xml from intersection.json
│   ├── data_to_rou.py               ← Standalone demand generator (from SUMO-Demand-Generation-Pipeline)
│   └── generate_edge_mapping.py     ← Geometry-based edge inspector (from SUMO-Demand-Generation-Pipeline)
│
├── modelling/
│   ├── agent.py                     ← Composes components; runs one step of the RL loop
│   ├── trainer.py                   ← Epoch loop; saves checkpoints + logs
│   └── components/
│       ├── environment/sumo_environment.py   ← SUMO/TraCI wrapper
│       ├── observation/queue_observation.py  ← Builds state vector from lane queue counts
│       ├── reward/wait_time.py               ← Reward = negative halting vehicle count
│       ├── policy/dqn.py                     ← DQN with online/target networks, epsilon-greedy
│       └── replay_buffer/uniform.py          ← Fixed-capacity circular buffer
│
├── baselines/                       ← NEW — rule-based baseline controllers (do NOT modify modelling/)
│   ├── __init__.py                  ← Exports FixedTimePolicy and ActuatedPolicy
│   ├── fixed_time.py                ← FixedTimePolicy — switches on a fixed timer, ignores queues
│   └── actuated.py                  ← ActuatedPolicy  — switches when queue clears or max green hit
│
└── Testcases/
    ├── Test_BuildNetwork.py
    └── Test_BuildRoute.py
```

---

## What has been completed

### 1. Full RL training pipeline — DONE
- DQN agent trained for 3 epochs across 50 training days
- 150 total training episodes, checkpoints saved every 10 episodes
- Final model saved to `src/data/models/final_model.pt`

### 2. DQN evaluation on test days — DONE
Results are already saved in `src/data/models/test_log.json`:

| Day ID | Total Reward | Steps |
|--------|-------------|-------|
| 1      | -58.06      | 252   |
| 7      | -61.69      | 252   |
| 15     | -61.25      | 252   |
| 17     | -59.13      | 252   |
| 40     | -46.50      | 252   |
| 47     | -46.44      | 252   |

**DQN mean test reward: -55.52** across 6 test days.
Epsilon at evaluation time: 0.55 (agent was still exploring — only 3 epochs trained).

### 3. Preprocessing pipeline — DONE
- 56 days of synthetic Toronto traffic data processed
- SUMO flow files generated for all 56 days
- McCowan_Finch.net.xml built and post-processed

### 4. Baseline policies written — DONE (but not yet evaluated)
Two rule-based controllers have been implemented in `baselines/`:

**`FixedTimePolicy`** (`baselines/fixed_time.py`)
- Holds each green phase for exactly `fixed_green_steps` decision steps then switches
- Does NOT look at queue state — timer only
- Represents the simplest possible controller (floor baseline)

**`ActuatedPolicy`** (`baselines/actuated.py`)
- Holds green while `mean_queue >= queue_threshold`
- Switches early when queue drains, or forces switch at `max_green_steps`
- Reads queue data from the observation vector (`obs[0:max_lanes]`)
- Represents a competent hand-crafted heuristic (harder baseline)

Both policies:
- Implement the same interface as `DQNPolicy` (`select_action`, `update`, `save`, `load`, `reset_phase_tracking`, `epsilon`)
- Are drop-in replacements inside the existing `Agent` class
- Have `update()` → `None` (no learning) and `save()`/`load()` → JSON config file
- Do NOT modify any existing `modelling/` code

### 5. SUMO-Demand-Generation-Pipeline merged — DONE
The standalone pipeline repo was merged into `preprocessing/`:
- `data_to_rou.py` and `generate_edge_mapping.py` copied in
- `BuildRoute.py` enhanced with `normalize_col()`, `--date-mode`, `--write-sumocfg`
- `BuildNetwork.py` enhanced with `write_edge_mapping()` auto-generating `edge_mapping.csv`

---

## What still needs to be done

### STEP 1 — Create `evaluate_baseline.py`  ← DO THIS FIRST

This script does not exist yet. It needs to:

1. Accept the same CLI args as `main.py`: `--config`, `--intersection`, `--sumo-home`
2. Build `SumoEnvironment`, `QueueObservation`, `WaitTimeReward` from config values
3. For each baseline policy:
   - Instantiate the policy with the correct parameters (see parameter table below)
   - Create `Agent(environment, observation, reward, policy, replay_buffer)`
   - Create `Trainer(agent, split_path, flows_dir, output_dir, n_epochs=0)`
   - Call `trainer.run()` — setting `n_epochs=0` skips training and runs only the test-day evaluation loop
   - Save log to `src/data/models/baseline_fixed_time_log.json` / `baseline_actuated_log.json`
4. Load all three logs and print a comparison table

**Key insight:** `n_epochs=0` makes the existing `Trainer` skip its training loop entirely
(`for epoch in range(1, 0+1)` → empty range) and jump straight to the test evaluation.
No new evaluation infrastructure is needed.

**Parameter values to use** (derived from `config.json` + `intersection.json`):

| Parameter | Value | Source |
|-----------|-------|--------|
| `min_green_steps` | `15` | `intersection.json → min_green_s` (same value DQN uses) |
| `max_green_steps` | `90` | `intersection.json → max_green_s` (same value DQN uses) |
| `fixed_green_steps` | `30` | Midpoint — one switch every 30 decision steps |
| `queue_threshold` | `0.1` | ~2 vehicles per lane out of max_vehicles=20 |
| `max_lanes` | `16` | `config.json → observation → max_lanes` |
| `decision_gap` | `30` | `config.json → simulation → decision_gap` |
| `step_length` | `5.0` | `config.json → simulation → step_length` |

**Paths to use:**

```python
split_path = "src/data/processed/split.json"
flows_dir  = "src/data/sumo/flows"
net_file   = "src/data/sumo/network/McCowan_Finch.net.xml"
models_dir = "src/data/models"
```

---

### STEP 2 — Run the evaluation

SUMO must be installed. On Windows the default path is:
`C:\Program Files (x86)\Eclipse\Sumo`

```bash
python evaluate_baseline.py \
    --config        config.json \
    --intersection  src/intersection.json \
    --sumo-home     "C:\Program Files (x86)\Eclipse\Sumo"
```

This will run the 6 test days twice (once per baseline) in SUMO — expect ~10–20 minutes.
The DQN model does NOT need to be re-run; its results are already in `test_log.json`.

---

### STEP 3 — Plot the results

Once all three log files exist, create a `plot_results.py` (or a Jupyter notebook)
that produces these charts for the capstone report:

1. **Bar chart** — mean total reward per model (Fixed-Time, Actuated, DQN)
2. **Box plot** — reward distribution across the 6 test days per model
3. **Line chart** — reward per test day (day ID on x-axis) for all three models overlaid
4. **Learning curve** — DQN training reward over 150 episodes (already in `train_log.json`)

The three log files all use the same schema:
```json
[
  {
    "total_reward": float,
    "steps":        int,
    "mean_loss":    float or null,
    "day_id":       int
  },
  ...
]
```
`mean_loss` is `null` for both baselines (no learning). Use `total_reward` and `day_id`
for all plots.

---

### STEP 4 — Consider more DQN training (optional)

The current DQN was only trained for 3 epochs (150 episodes) and epsilon was still
**0.55** at evaluation time (agent was still half-random). The model has not converged.

Options:
- Re-run `main.py` with `n_epochs = 10` or `20` in `config.json` to get a properly
  trained agent before the final comparison
- Or keep the current results and discuss underfitting as a limitation in the report

---

## Key simulation parameters

| Setting | Value | Meaning |
|---------|-------|---------|
| `step_length` | 5.0 s | One SUMO simulation tick = 5 seconds |
| `decision_gap` | 30 ticks | Agent makes a decision every 30 ticks |
| 1 decision step | 150 s | 30 × 5 s = 2.5 minutes of real traffic |
| `begin` | 27000 s | Simulation starts at 07:30 AM |
| `end` | 64800 s | Simulation ends at 06:00 PM |
| Total steps/episode | 252 | (64800 − 27000) ÷ 150 = 252 |
| Reward | negative halting count | Normalised by lane count, averaged over all TLS |

---

## Important design decisions (don't change without reading rationale)

- **Baselines live in `baselines/`** — completely separate from `modelling/` so the
  existing RL code is never touched. Both baselines import from `modelling/` but do
  not modify it.
- **`n_epochs=0` trick** — passing `n_epochs=0` to `Trainer` skips training and runs
  only the test evaluation. This reuses all existing infrastructure without new code.
- **Reward = negative halting count** — `WaitTimeReward` uses `getLastStepHaltingNumber`
  (instantaneous), not cumulative waiting time. This gives a clean signal each step.
- **Action space is binary** — `0 = keep phase`, `1 = advance to next phase`. Both
  baselines and DQN use the same two-action space.
- **Min/max green values** — `min_green_steps=15`, `max_green_steps=90` are passed
  directly from `intersection.json` into all three policies. This keeps the constraints
  identical for a fair comparison.

---

## Previous conversation context

This project was built across multiple Cursor sessions:

1. **Session 1** — RL training pipeline built and run; DQN trained and evaluated;
   all `src/data/` artifacts generated.
2. **Session 2** — `SUMO-Demand-Generation-Pipeline` repo merged into `preprocessing/`;
   `BuildRoute.py` and `BuildNetwork.py` enhanced; `PIPELINE_OVERVIEW.md` created.
3. **Session 3** (this session) — `baselines/fixed_time.py` and `baselines/actuated.py`
   created as standalone rule-based controllers; `HANDOFF.md` written.

**The machine used for sessions 1–3 does not have SUMO installed.**
The next session should be on a machine with SUMO to run the baseline evaluation.
