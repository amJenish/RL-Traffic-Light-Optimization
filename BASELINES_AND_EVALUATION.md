# Baselines and Evaluation

This document covers the two rule-based baseline controllers, the script that evaluates them, and the script that plots all results.

---

## `baselines/fixed_time.py` — Fixed-Time Policy

### What it is

The Fixed-Time policy is the simplest possible traffic signal controller. It switches the green phase on a fixed timer and completely ignores what is actually happening on the road. No sensors, no queue data — just a clock.

This is a useful baseline because it represents how most real-world signals are operated by default ("optimise once offline, then leave it"). Comparing DQN against it answers the question: *is reacting to live queue state worth the added complexity?*

### How it works

The policy tracks how many decision steps have passed since the last phase switch. At each step it applies the following logic in order:

| Condition | Action |
|---|---|
| Steps ≥ `max_green_steps` | Force switch (prevents infinite green) |
| Steps < `min_green_steps` | Block switch (prevents rapid oscillation) |
| Steps ≥ `fixed_green_steps` | Timer fired — switch |
| Otherwise | Hold current phase |

`fixed_green_steps` is the target hold duration (default: 30 steps × 5 s × 30 = 150 s per phase). `min_green_steps` and `max_green_steps` are safety bounds derived from `intersection.json` and are identical to the bounds enforced on the DQN agent, ensuring a fair comparison.

### Key parameters

| Parameter | Default | Source |
|---|---|---|
| `fixed_green_steps` | 30 | Hardcoded in `evaluate_baseline.py` |
| `min_green_steps` | 1 | `intersection.json` → `min_green_s` |
| `max_green_steps` | 3 | `intersection.json` → `max_green_s` |

### Interface compatibility

`FixedTimePolicy` extends `BasePolicy` so it can be dropped into the existing `Agent`/`Trainer` pipeline without any changes. It implements `update()` as a no-op (it does not learn), `save()`/`load()` as JSON config files (no neural network weights), and `epsilon` returns `None` (no exploration).

---

## `baselines/actuated.py` — Actuated Policy

### What it is

The Actuated policy is the standard "smart but not learning" approach used in modern real-world traffic engineering. Unlike Fixed-Time, it reads the current queue length from the observation vector and reacts accordingly. It represents the best a well-designed hand-crafted rule can do.

Comparing DQN against Actuated answers the harder question: *is learned behaviour better than a competent deterministic heuristic?*

### How it works

Each decision step, the policy reads the first `max_lanes` elements of the observation vector — the normalised halting-vehicle counts per lane (0.0 = empty lane, 1.0 = fully blocked lane). It computes the mean queue across all active lanes (lanes reporting at least one vehicle), then applies this logic in priority order:

| Priority | Condition | Action |
|---|---|---|
| 1 | Steps ≥ `max_green_steps` | Force switch |
| 2 | Steps < `min_green_steps` | Block switch |
| 3 | Mean queue < `queue_threshold` | Queue has drained — switch |
| 4 | Otherwise | Queue still active — hold green |

The idea is: keep the green phase running while vehicles are still queued, then switch as soon as the intersection drains. If traffic never clears, the `max_green_steps` cap forces a switch so the opposing direction is not starved.

### Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_lanes` | 16 | Must match `QueueObservation.max_lanes` in `config.json` |
| `min_green_steps` | 1 | Hard lower bound on phase duration |
| `max_green_steps` | 3 | Hard upper bound on phase duration |
| `queue_threshold` | 0.1 | ~2 vehicles per lane (with `max_vehicles=20`) |

### Interface compatibility

Same as `FixedTimePolicy` — extends `BasePolicy`, `update()` is a no-op, no neural network weights, `epsilon` returns `None`.

---

## `evaluate_baseline.py` — Baseline Evaluation Script

### What it does

Runs both baseline policies through the exact same test-day evaluation loop that the DQN uses, then prints a side-by-side comparison table and saves the results as JSON logs.

### The `n_epochs=0` trick

The script reuses the existing `Trainer` class by passing `n_epochs=0`. This skips the training loop entirely and runs only the held-out test evaluation. No new evaluation infrastructure is needed — the baselines are evaluated identically to how the DQN is evaluated after training.

### Step-by-step flow

1. Parse CLI arguments (`--config`, `--intersection`, `--sumo-home`).
2. Load `config.json` and `intersection.json`.
3. Instantiate `FixedTimePolicy` and `ActuatedPolicy` with parameters derived from those files.
4. For each policy, call `_run_baseline()`:
   - Builds a fresh `Agent` (environment, observation, reward, replay buffer) wired to the policy.
   - Creates a `Trainer` with `n_epochs=0` and a temporary output directory so the DQN logs are never overwritten.
   - Calls `trainer.run()` and returns the `test_log`.
5. Save both logs to `src/data/models/`:
   - `baseline_fixed_time_log.json`
   - `baseline_actuated_log.json`
6. Load `test_log.json` (DQN results) and print a three-way comparison table.

### Output files

| File | Contents |
|---|---|
| `src/data/models/baseline_fixed_time_log.json` | One entry per test day: `total_reward`, `steps`, `mean_loss` (null), `day_id` |
| `src/data/models/baseline_actuated_log.json` | Same schema |
| `src/data/models/_tmp_fixed_time/` | Temporary Trainer artefacts (safe to ignore) |
| `src/data/models/_tmp_actuated/` | Temporary Trainer artefacts (safe to ignore) |

### How to run

```bash
python evaluate_baseline.py \
    --config       config.json \
    --intersection src/intersection.json \
    --sumo-home    "C:\Program Files (x86)\Eclipse\Sumo"
```

---

## `plot_results.py` — Results Plotter

### What it does

Reads the three test logs and the training log, then generates four publication-ready PNG charts saved to `src/data/models/plots/`.

### The four charts

**`bar_mean_reward.png`** — Bar chart showing the mean total reward across all 6 test days for each policy. Higher (less negative) is better. Useful for a quick headline comparison.

**`box_reward_dist.png`** — Box plot showing the distribution (median, IQR, whiskers) of per-day rewards for each policy. Shows which policy is most consistent across different traffic days.

**`line_reward_per_day.png`** — All three policies plotted on the same line chart with test day ID on the x-axis. Shows which days were hard or easy for all policies equally, and whether any policy has an edge on specific days.

**`learning_curve.png`** — DQN training reward over 150 episodes. Raw per-episode reward shown in grey; a 10-episode rolling mean shown in green to reveal the underlying trend. A flat or declining curve indicates the model needs more training.

### Log file schema

All three test logs share the same structure:

```json
[
  {
    "total_reward": float,
    "steps":        int,
    "mean_loss":    float or null,
    "day_id":       int
  }
]
```

`mean_loss` is `null` for both baselines since they do not learn. The training log additionally includes `episode`, `epoch`, and `epsilon` fields.

### How to run

```bash
python plot_results.py
```

Must be run from the project root. Overwrites the PNGs each time, so re-run after any new training or baseline evaluation to refresh the charts.

### Regenerating after more training

After increasing `n_epochs` in `config.json` and retraining, run all three scripts in sequence:

```bash
python main.py
python evaluate_baseline.py --config config.json --intersection src/intersection.json --sumo-home "C:\Program Files (x86)\Eclipse\Sumo"
python plot_results.py
```
