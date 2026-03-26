# Web UI (Streamlit)

This repo uses **[Streamlit](https://docs.streamlit.io)** for the simplest, most-documented stack: **one language (Python)**, no separate Node/React build, and official tutorials for file upload, dataframes, and charts.

## Run

From the project root (with your virtualenv activated):

```bash
pip install streamlit   # or: pip install -r requirements.txt
python run_frontend.py
```

Equivalent direct command:

```bash
streamlit run streamlit_app.py
```

The browser opens automatically. Set **`SUMO_HOME`** in the sidebar (or install `eclipse-sumo` via pip and paste the path printed by `python -c "import sumo; print(sumo.SUMO_HOME)"`).

## What it does

1. **Upload** a traffic CSV each session (format similar to `src/data/synthetic_toronto_data.csv`; that file is not auto-loaded).
2. **Generate from CSV** fills the column map and intersection in memory; refine with the **field editors** (sections 2–3) or under **Raw JSON** (paste / copy). See [STREAMLIT_ARCHITECTURE.md](STREAMLIT_ARCHITECTURE.md) for the full data flow.
3. Open **Advanced** to change epochs, train/test days, simulation window, etc. (these map to `main.py` constants).  
   **Default EPOCHS in the UI** is **`STREAMLIT_DEFAULT_EPOCHS`** in [`streamlit_app.py`](streamlit_app.py) (set to **10** for quick tests). Change that constant to `60` to match `main.py`, or type any value in Advanced.
4. **Run training pipeline** — calls `validate_inputs` → `run_build_route` → `run_build_network` → `build_pipeline` → `trainer.run()` from [`main.py`](main.py). Outputs land under `logs/<timestamp>_.../` like the CLI.
5. **Evaluate baselines** — runs [`evaluate_baseline.py`](evaluate_baseline.py) in a subprocess. If you trained in the same session, DQN results are picked up from the last run folder for the comparison table in the script output.

Local scratch files are stored under **`streamlit_runs/`** (gitignored).

## Docs

- [Streamlit get started](https://docs.streamlit.io/get-started)
- [File upload](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader)
- [Dataframes & charts](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)

## Limitations (by design)

- Training runs **in the Streamlit process**; keep the tab open until it finishes. For very long runs, prefer `python main.py` in a terminal.
- Policy/reward class selection still follows **`main.py`** imports (not changed from the UI in this minimal version).
