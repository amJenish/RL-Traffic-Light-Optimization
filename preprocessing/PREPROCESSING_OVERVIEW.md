# Preprocessing Overview

## Files Removed

Two standalone scripts from the `SUMO-Demand-Generation-Pipeline` repo were
removed from `preprocessing/` because their functionality was fully absorbed
into `BuildRoute.py` and `BuildNetwork.py`:


| Removed file               | Why removed                                                                                                                                                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_to_rou.py`           | Its core logic (column normalisation, multi-date handling, `.xls` loading, SUMO config writing, XML route output) was ported into `BuildRoute.py`. It was never called in the pipeline — `BuildRoute.py` handles all route generation. |
| `generate_edge_mapping.py` | Its `write_edge_mapping()` function was ported into `BuildNetwork.py` and now runs automatically every time the network is built. The standalone script was never called in the pipeline.                                              |


---

## How `BuildNetwork.py` and `BuildRoute.py` Cover Everything

### `BuildNetwork.py` — intersection network builder

Takes `intersection.json` and produces the complete SUMO road network:

1. Writes junction positions (`.nod.xml`)
2. Writes inbound and outbound edges for each approach (`.edg.xml`)
3. Writes lane-to-lane turn connections (`.con.xml`)
4. Writes traffic light phase definitions (`.tll.xml`)
5. Calls `netconvert` to assemble the final `McCowan_Finch.net.xml`
6. Post-processes phase durations to enforce `min_green` / `max_green` from `intersection.json`
7. **Automatically writes `edge_mapping.csv`** — this was the only unique feature of `generate_edge_mapping.py`. It now runs as part of every network build, so no separate step is needed.

All edges follow the naming convention `edge_{D}_in` / `edge_{D}_out` (e.g.
`edge_N_in`, `edge_E_out`), which means the edge mapping never needs geometry
parsing — it is derived directly from the config.

---

### `BuildRoute.py` — demand and split generator

Takes the traffic count CSV, `intersection.json`, and `columns.json`, and
produces all inputs the RL trainer needs:

1. **Loads the CSV** — handles `.csv`, `.xls`, and `.xlsx`; normalises column
  name casing and spacing so mismatched headers never cause failures
   (`normalize_col()`, ported from `data_to_rou.py`)
2. **Assigns day indices** — supports `--date-mode concat/offset/error` for
  multi-date input (ported from `data_to_rou.py`)
3. **Writes per-day CSVs** → `processed/days/day_NN.csv`
4. **Writes SUMO flow files** → `sumo/flows/flows_day_NN.rou.xml`, one per day,
  using `vehsPerHour` for continuous vehicle injection
5. **Writes `split.json`** — randomised train/test day split consumed by the
  `Trainer`
6. **Optionally writes `simulation.sumocfg`** — lets you open any day in
  `sumo-gui` for visual inspection (`--write-sumocfg` flag, ported from
   `data_to_rou.py`)

---

## Current `preprocessing/` Contents

```
preprocessing/
├── BuildNetwork.py            ← Build SUMO network + edge_mapping.csv
├── BuildRoute.py              ← Build per-day flow files + split.json
└── PREPROCESSING_OVERVIEW.md ← This file
```

---

## Typical Usage

```bash
# 1. Build the SUMO network (also generates edge_mapping.csv)
python preprocessing/BuildNetwork.py \
    --config src/intersection.json \
    --out-dir src/data/sumo/network

# 2. Build per-day route files and train/test split
python preprocessing/BuildRoute.py \
    --csv           src/data/synthetic_toronto_data.xls \
    --intersection  src/intersection.json \
    --columns       src/columns.json \
    --out-dir       src/data

# 3. Run the full RL pipeline (calls both of the above automatically)
python main.py \
    --csv           src/data/synthetic_toronto_data.xls \
    --intersection  src/intersection.json \
    --columns       src/columns.json \
    --sumo-home     "C:\Program Files (x86)\Eclipse\Sumo"
```

