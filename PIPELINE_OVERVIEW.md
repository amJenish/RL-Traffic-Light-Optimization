# RL Traffic Light Optimization — Pipeline Overview

This document describes how the project is structured, how the two source
repositories were merged, and how every piece connects end-to-end.

---

## Repository origins

| Repository | Role |
|---|---|
| `RL-Traffic-Light-Optimization` | Full RL training pipeline (preprocessing → simulation → DQN agent → evaluation) |
| `SUMO-Demand-Generation-Pipeline` | Standalone demand-generation utility (count CSV → SUMO `rou.xml`) |

The standalone pipeline has been merged **into** this repo under `preprocessing/`
so everything needed to reproduce a run lives in one place.

---

## Merge mapping table

The table below shows exactly which feature came from which repo, where it now
lives, and what was changed.

| Feature | Origin repo / file | Lives in | Change |
|---|---|---|---|
| `normalize_col()` — case/spacing-tolerant column matching | `SUMO-Demand-Generation-Pipeline/data_to_rou.py` | `preprocessing/BuildRoute.py` | Copied as helper; `load_csv()` now normalizes both CSV headers and `columns.json` values before matching |
| `.xls` / `.xlsx` loading | `SUMO-Demand-Generation-Pipeline/data_to_rou.py` | `preprocessing/BuildRoute.py` | `load_csv()` now reads Excel files in addition to `.csv` |
| `--date-mode` (error / offset / concat) | `SUMO-Demand-Generation-Pipeline/data_to_rou.py` | `preprocessing/BuildRoute.py` | `assign_sim_days()` now accepts a `date_mode` parameter; `main()` exposes `--date-mode` flag |
| `--write-sumocfg` + `write_sumocfg()` | `SUMO-Demand-Generation-Pipeline/data_to_rou.py` | `preprocessing/BuildRoute.py` | New function writes a `.sumocfg` for `sumo-gui` visual inspection after route files are generated |
| `write_edge_mapping()` | `SUMO-Demand-Generation-Pipeline/generate_edge_mapping.py` | `preprocessing/BuildNetwork.py` | Generates `edge_mapping.csv` automatically after `netconvert`; uses known `edge_{D}_in / edge_{D}_out` naming — no geometry parsing needed |
| `data_to_rou.py` (standalone demand tool) | `SUMO-Demand-Generation-Pipeline/data_to_rou.py` | `preprocessing/data_to_rou.py` | Copied as-is; works directly with `synthetic_toronto_data.xls` + the `edge_mapping.csv` that `BuildNetwork.py` now auto-generates |
| `generate_edge_mapping.py` (geometry-based edge detection) | `SUMO-Demand-Generation-Pipeline/generate_edge_mapping.py` | `preprocessing/generate_edge_mapping.py` | Copied as-is; useful for detecting edges on any arbitrary SUMO network |
| `requirements.txt` | `SUMO-Demand-Generation-Pipeline/requirements.txt` | `requirements.txt` (repo root) | Merged with RL dependencies (`torch`, `numpy`) into a single file |

---

## High-level architecture

```
synthetic_toronto_data.xls / .csv
        │
        │  (columns.json maps column names)
        ▼
┌─────────────────────────────────────────────────────┐
│  preprocessing/BuildRoute.py                        │
│  ─────────────────────────────────────────────────  │
│  • Loads & normalises CSV headers (case-insensitive) │
│  • Assigns sim_day indices (--date-mode)             │
│  • Writes per-day CSVs  → processed/days/day_NN.csv  │
│  • Writes SUMO flows    → sumo/flows/flows_day_NN.rou.xml │
│  • Writes train/test split → processed/split.json   │
│  • Optionally writes simulation.sumocfg (--write-sumocfg) │
└─────────────────────────────────────────────────────┘
        │
        │  intersection.json defines geometry & phase config
        ▼
┌─────────────────────────────────────────────────────┐
│  preprocessing/BuildNetwork.py                      │
│  ─────────────────────────────────────────────────  │
│  • Generates .nod.xml, .edg.xml, .con.xml, .tll.xml │
│  • Calls netconvert → McCowan_Finch.net.xml          │
│  • Post-processes TLS phase durations               │
│  • NEW: writes edge_mapping.csv (merged from        │
│         generate_edge_mapping.py)                   │
└─────────────────────────────────────────────────────┘
        │                          │
        │ (net.xml + flow files)   │ (edge_mapping.csv — optional path)
        ▼                          ▼
┌───────────────────┐   ┌──────────────────────────────┐
│  RL Training      │   │  preprocessing/data_to_rou.py │
│  pipeline         │   │  ─────────────────────────── │
│  (main.py)        │   │  Standalone demand tool from  │
│                   │   │  SUMO-Demand-Generation-      │
│  SumoEnvironment  │   │  Pipeline. Converts any count │
│  QueueObservation │   │  CSV → rou.xml using          │
│  WaitTimeReward   │   │  edge_mapping.csv or auto-    │
│  DQNPolicy        │   │  edge detection on net.xml.   │
│  UniformReplay    │   │  Uses vehicle counts (number=)│
│  Agent + Trainer  │   │  rather than vehsPerHour.     │
└───────────────────┘   └──────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  src/data/models/                                   │
│  • checkpoint_epNNNN.pt  (periodic saves)           │
│  • final_model.pt                                   │
│  • train_log.json   (reward, loss, epsilon/episode) │
│  • test_log.json    (held-out day evaluation)       │
└─────────────────────────────────────────────────────┘
```

---

## File-by-file reference

### `preprocessing/` — data preparation

| File | Purpose |
|---|---|
| `BuildRoute.py` | **Primary route builder for RL.** Reads count CSV + config files, outputs per-day CSVs, SUMO flow files (using `vehsPerHour`), and `split.json`. Enhanced with `normalize_col`, `--date-mode`, and `--write-sumocfg`. |
| `BuildNetwork.py` | **Intersection network builder.** Generates all SUMO network XML files and calls `netconvert`. Now also writes `edge_mapping.csv` automatically. |
| `data_to_rou.py` | **Standalone demand generator** (from pipeline repo). Takes any count CSV → `rou.xml` using absolute vehicle counts (`number=`). Best for ad-hoc experiments or checking demand without running the full RL pipeline. |
| `generate_edge_mapping.py` | **Geometry-based edge inspector** (from pipeline repo). Parses any `net.xml`, detects the central junction, and assigns N/S/E/W edge IDs by angle. Useful if you swap in a different SUMO network. |

### `modelling/` — RL components

| File | Purpose |
|---|---|
| `agent.py` | Composes all components; runs one step: select action → apply → advance sim → collect reward → store transition → update policy. |
| `trainer.py` | Epoch loop over training days + final evaluation on held-out test days. Saves checkpoints and JSON logs. |
| `components/environment/sumo_environment.py` | SUMO/TraCI wrapper. Manages episode lifecycle (start → step → done → close). |
| `components/observation/queue_observation.py` | Builds the state vector: halting counts per lane + phase info + time-of-day and day-of-week sin/cos encoding. |
| `components/reward/wait_time.py` | Instantaneous negative halting-vehicle count. Goes down when queues grow, up when they clear — clean signal for DQN. |
| `components/policy/dqn.py` | Online + target Q-networks (2-layer MLP), Adam optimiser, epsilon-greedy with multiplicative decay, hard target sync every N updates, min/max green phase enforcement. |
| `components/replay_buffer/uniform.py` | Fixed-capacity circular buffer, uniform random sampling. |

### Top-level

| File | Purpose |
|---|---|
| `main.py` | Single entry point. Validates inputs → builds routes → builds network → constructs components → runs training + evaluation. |
| `config.json` | All hyperparameters (DQN, simulation window, observation, replay buffer, output paths). |
| `src/intersection.json` | Intersection geometry: approach lane counts, speed, phase mode, timing constraints. |
| `src/columns.json` | Maps CSV column names to the standard names used by `BuildRoute.py`. |
| `requirements.txt` | All Python dependencies (merged from both repos). |

---

## Two route generation tools — why both exist

`BuildRoute.py` and `data_to_rou.py` solve the same core problem in slightly
different ways. They are complementary, not redundant.

| | `BuildRoute.py` | `data_to_rou.py` |
|---|---|---|
| **Designed for** | RL training loop | Ad-hoc / standalone SUMO runs |
| **Demand format** | `vehsPerHour` — continuous flow rate | `number` — fixed vehicle count per window |
| **Multi-day output** | One `rou.xml` per day (Trainer iterates over them) | Single concatenated `rou.xml` (all days in one file) |
| **Column config** | `columns.json` — arbitrary column names | Normalized `n_approaching_t` pattern |
| **Edge IDs** | Auto-named `edge_{D}_in/out` (from `BuildNetwork.py`) | Requires `edge_mapping.csv` or `--auto-edges` |
| **Train/test split** | Yes — `split.json` | No |
| **SUMO config** | `--write-sumocfg` (optional) | `--write-sumocfg` (optional) |

**Typical workflow:**
1. Run `BuildNetwork.py` → get `net.xml` + `edge_mapping.csv`
2. Run `BuildRoute.py` → get per-day flow files for RL training
3. (Optional) Run `data_to_rou.py` with `edge_mapping.csv` → get a single
   `rou.xml` to visually inspect any day in `sumo-gui`

---

## Quick-start commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the SUMO network (also generates edge_mapping.csv)
python preprocessing/BuildNetwork.py \
    --config src/intersection.json \
    --out-dir src/data/sumo/network

# 3. Build routes + split (add --write-sumocfg to get a sumo-gui config)
python preprocessing/BuildRoute.py \
    --csv            src/data/synthetic_toronto_data.csv \
    --intersection   src/intersection.json \
    --columns        src/columns.json \
    --out-dir        src/data \
    --date-mode      concat \
    --write-sumocfg

# 4. (Optional) Inspect demand in sumo-gui
sumo-gui -c simulation.sumocfg

# 5. Run full RL pipeline
python main.py \
    --csv           src/data/synthetic_toronto_data.xls \
    --intersection  src/intersection.json \
    --columns       src/columns.json \
    --sumo-home     "/path/to/sumo"
```
