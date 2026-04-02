# Front-end

Visualization scripts and optional static assets for the RL traffic project. This folder is **decoupled** from the Python training pipeline: it does not import `main`, `modelling`, or TraCI.

## Configuration

Paths and training-related settings come from the repo root **`config.json`** (see `output.out_dir`, `output.models_dir`, `output.logs_dir`, `training`, `simulation`, etc.). Edit that file to change where logs and models are written; the scripts here read the same file.

Run scripts from the **repository root** so `config.json` resolves correctly:

```bash
python front-end/scripts/plot_results.py
python front-end/scripts/dqn_graphs.py
```

## Layout

| Path | Purpose |
|------|---------|
| `scripts/config_paths.py` | Resolves repo root and directories from `config.json` |
| `scripts/plot_results.py` | Baseline vs DQN comparison plots → `models_dir/plots/` |
| `scripts/dqn_graphs.py` | Training curves grid → `out_dir/graphs/` |
| `public/` | Placeholder for a future static web UI (see `public/README.md`) |

## Prerequisites

- Python with `matplotlib`, `numpy` (same environment as training).
- Log files under `output.models_dir` from training / `evaluate_baseline.py` where applicable.
