# RL Grid Search (GridSearchCV-style)

This folder contains a simple RL hyperparameter sweep runner for your traffic-signal DQN pipeline.

It is **not** sklearn’s `GridSearchCV`; it’s a practical “parameter sweep”:
- iterate over combinations of reward + policy parameters from JSON search spaces
- run each trial end-to-end on train days
- evaluate on test days
- compute a shared KPI (**crossings**) that matches `ThroughputReward` semantics
- write `results.csv` and `leaderboard.csv` under the dedicated grid-search run folder

## Files

- `grid_search/gridsearch.py`
- `reward_search_space.json`
- `policy_search_space.json`
- `runs/<timestamp>/...` (generated)

## How crossings KPI is computed

The leaderboard ranks trials using **crossings counts exactly like** `modelling/components/reward/throughput.py`:
- it counts newly-seen vehicle IDs on the counted departure lanes
- it follows the same decision-interval timing as your `Agent` (SMDP-style)

So `reward_configuration.json` can be `CompositeReward`, `ThroughputQueueReward`, etc.—the leaderboard metric is still the same crossings KPI.

## Usage

From the repository root:

```bash
python experiments/grid_search/gridsearch.py
```

Common options:

```bash
python experiments/grid_search/gridsearch.py --max_trials 10
python experiments/grid_search/gridsearch.py --use_gui
python experiments/grid_search/gridsearch.py --output_root experiments/grid_search
```

## JSON search spaces

### `reward_search_space.json`
Expected shape:

```json
{
  "rewards": [
    {
      "RewardClass": "ThroughputQueueReward",
      "params": { "queue_weight": [0.5, 1.0], "throughput_weight": [1.0] }
    }
  ]
}
```

### `policy_search_space.json`
Expected shape:

```json
{
  "policies": [
    {
      "PolicyClass": "DoubleDQNPolicy",
      "seeds": [42, 123],
      "params": { "lr": [0.0005, 0.001], "gamma": [0.97] }
    }
  ]
}
```

## Interpreting results

Inside each trial folder you’ll find:
- `reward_configuration.json` (snapshot for that trial)
- `policy_configuration.json` (snapshot for that trial)
- `config.txt` (trial summary)
- `train_log.json`, `test_log.json`
- `graphs/` — `train_curves.png` and `test_rewards.png` (written automatically when the trial finishes, same plots as `visualization.render_run_graphs`)
- `checkpoints/` and `final_model.pt` (if the trial runs)

The sweep-level files:
- `results.csv`: one row per trial (per seed)
- `leaderboard.csv`: aggregated per config key (seed-aggregated), sorted by:
  - `mean_crossings_rate` (descending)
  - `std_crossings_rate` (ascending)

