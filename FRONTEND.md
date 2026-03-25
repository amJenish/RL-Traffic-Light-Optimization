# Web UI (Streamlit)

This repo uses **[Streamlit](https://docs.streamlit.io)** for the simplest, most-documented stack: **one language (Python)**, no separate Node/React build, and official tutorials for file upload, dataframes, and charts.

## Run

From the project root (with your virtualenv activated):

```bash
pip install streamlit   # or: pip install -r requirements.txt
streamlit run streamlit_app.py
```

The browser opens automatically. Set **`SUMO_HOME`** in the sidebar (or install `eclipse-sumo` via pip and paste the path printed by `python -c "import sumo; print(sumo.SUMO_HOME)"`).

## What it does

1. **Upload** a traffic CSV (same idea as `src/data/synthetic_toronto_data.csv`).
2. Edit **`columns.json`** then **`intersection.json`** (defaults load from `src/`). Order matters so the suggestion helper can read your column map.
3. **Suggest intersection from CSV** (expander) — detects active approaches from `columns.json` or from `n_approaching_*` column names; you set **lane counts**, **speed**, and optional **timing**; applies defaults from [`streamlit_intersection_helpers.py`](streamlit_intersection_helpers.py). **Apply auto columns.json** guesses a column map from Toronto-style headers. You can still edit every JSON field by hand.
4. Open **Advanced** to change epochs, train/test days, simulation window, etc. (these map to `main.py` constants).  
   **Default EPOCHS in the UI** is **`STREAMLIT_DEFAULT_EPOCHS`** in [`streamlit_app.py`](streamlit_app.py) (set to **10** for quick tests). Change that constant to `60` to match `main.py`, or type any value in Advanced.
5. **Run training pipeline** — calls `validate_inputs` → `run_build_route` → `run_build_network` → `build_pipeline` → `trainer.run()` from [`main.py`](main.py). Outputs land under `logs/<timestamp>_.../` like the CLI.
6. **Evaluate baselines** — runs [`evaluate_baseline.py`](evaluate_baseline.py) in a subprocess. If you trained in the same session, DQN results are picked up from the last run folder for the comparison table in the script output.

Local scratch files are stored under **`streamlit_runs/`** (gitignored).

## Docs

- [Streamlit get started](https://docs.streamlit.io/get-started)
- [File upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Dataframes & charts](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)

## Limitations (by design)

- Training runs **in the Streamlit process**; keep the tab open until it finishes. For very long runs, prefer `python main.py` in a terminal.
- Policy/reward class selection still follows **`main.py`** imports (not changed from the UI in this minimal version).
