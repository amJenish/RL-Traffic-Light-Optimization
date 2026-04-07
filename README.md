# RL Traffic Light Optimization

End-to-end pipeline for **reinforcement-learning-based traffic signal control** in **[SUMO](https://www.eclipse.org/sumo/)**: preprocess traffic-count data, build simulation networks and routes, train a DQN / Double DQN policy with TraCI, compare against fixed-time and actuated baselines, and inspect results via a **Streamlit** UI or the command line.

---

## Prerequisites

- **Python** 3.10+ (3.11 recommended)
- **[Eclipse SUMO](https://eclipse.dev/sumo/)** installed and available on your machine (the tools must run from `SUMO_HOME/bin`, e.g. `sumo`, `netconvert`)
- Optional: **CUDA**-capable GPU for faster training (PyTorch uses CPU if no GPU)

---

## Setup

From the **repository root**:

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Set **`SUMO_HOME`** to your SUMO installation root (the folder that contains `bin/sumo.exe` or `bin/sumo`), or configure it in the Streamlit sidebar / environment when you run the app.

---

## Run the front end (Streamlit)

The web UI drives the same pipeline as `main.py`: upload data, adjust options, run training, and download `schedule.json` and other artefacts.

1. Activate the virtualenv (see above).
2. From the **project root**:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Your browser should open the app. If not, use the URL printed in the terminal (usually `http://localhost:8501`).

4. In the sidebar, set **`SUMO_HOME`** if it is not already detected.

5. Typical flow: **upload a traffic count file (CSV / Excel)** → configure **intersection / column mapping** → optional **Advanced** (epochs, train/test days, simulation window, baseline comparison) → **Run** → open the **Results** tab for downloads and charts.

**Notes**

- Training runs **inside the Streamlit process**; keep the browser tab open until the run finishes. For long experiments, prefer `python main.py` in a separate terminal.
- Default training epochs in the UI are controlled by **`STREAMLIT_DEFAULT_EPOCHS`** in `streamlit_app.py` (tuned for quick demos). Use **Advanced** to set epochs to match batch training (e.g. values in `config.json`).
- Policy and reward **classes** are selected via `config.json` (`components.*`) and `main.py` registration, not from a full picker in the UI.
- Session scratch paths may use **`streamlit_runs/`** (see app behaviour); run outputs are written under the **results directory** in `config.json` (default: `src/data/results/`).

More UI-oriented detail is in [`FRONTEND.md`](FRONTEND.md).

---

## Run the pipeline from the CLI

From the project root, with `venv` activated and `SUMO_HOME` set:

```bash
python main.py
```

Optional:

```bash
python main.py --config path/to/config.json
python main.py --gui
```

Configuration defaults live in **`config.json`** at the repo root (paths, training, simulation, reward/policy blocks).

**Baselines** (fixed-time, actuated, Webster-style helpers as implemented):

```bash
python evaluate_baseline.py --help
```

---

## Repository layout (high level)

| Area | Role |
|------|------|
| `main.py` | CLI entry: validation, route/network build, RL training, schedules, optional replay/KPI |
| `streamlit_app.py` | Streamlit front end |
| `modelling/` | Agent, trainer, policies, rewards, replay buffer, schedules |
| `preprocessing/` | Route and network generation from data |
| `config.json` | Global defaults and component class names |
| `src/data/` | Sample data and default output locations (see `config.json`) |
| `evaluate_baseline.py` | Baseline evaluation script |
| `visualization/` | Plotting helpers for training/test logs |


