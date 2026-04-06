# Final report — old → new replacement guide

Use this document when updating **Final Report BACKUP.pdf** (or your current Word/LaTeX source).  
**Old** = delete or replace this. **New** = paste this. If **Old** is `empty`, there was no content in the extracted PDF (or insert at the indicated heading).

---

## 1. Abstract


|           |                                                                 |
| --------- | --------------------------------------------------------------- |
| **Where** | Section *Abstract* (page ~3 in BACKUP PDF)                      |
| **Old**   | `empty` (only the heading “Abstract” was present in extraction) |
| **New**   | See block below                                                 |


**New (Abstract — full text):**

Urban traffic signal control is often implemented with fixed or actuated rules that do not adapt fully to fluctuating demand. This project implements an end-to-end pipeline that converts turning-movement count data into SUMO demand, trains a **Double Deep Q-Network** agent with **ThroughputCompositeReward** in a **semi-Markov** setting, and evaluates the learned policy against classical baselines in the same simulator and time window (28,800–36,000 s). After **300 epochs** (five training days per epoch; **1,500** training episodes), evaluation on held-out test days showed a mean cumulative reward of **+1.61** per episode under the greedy policy, versus **−1,172.75** for the untrained random policy on the **same** test days (**+100.1%** improvement). On the three test days common to both the learned agent and the fixed-time and actuated baselines (days 0, 1, and 4), mean reward improved from **−1,551.19** and **−1,698.20** to **−18.51** (**~98.8%** and **~98.9%** improvement respectively). Training had not fully converged by the final epoch (**ε ≈ 0.49** at run end). The system exports logs, checkpoints, and a `**schedule.json`** timing summary. Results are simulation-based and limited to a **single** intersection.

---

## 2. Glossary


|           |                                                       |
| --------- | ----------------------------------------------------- |
| **Where** | Section *Glossary of Terms and Definitions* (page ~2) |
| **Old**   | `empty` (heading only in extraction)                  |
| **New**   | See block below                                       |


**New (Glossary — paste as bullet list or table):**

- **TraCI** — Traffic Control Interface; Python API to control SUMO during simulation.
- **SUMO** — Simulation of Urban MObility; microscopic traffic simulator.
- **SMDP** — Semi-Markov Decision Process; decisions at variable time intervals (green holds / phase switches).
- **Double DQN** — Deep Q-learning variant that reduces overestimation by decoupling action selection and value evaluation.
- **ε-greedy** — Exploration strategy: random action with probability ε, otherwise greedy action.
- **Epoch** — One pass over the configured set of training traffic days.
- **Episode** — One full simulation run on a single traffic day during training or evaluation.
- **ThroughputCompositeReward** — Scalar reward combining throughput-related terms with queue/wait penalties (weights α, β, γ in configuration).

---

## 3. Background — Webster baseline paragraph (Section 2)


|           |                                                                                                                                                                                                                                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | End of *Related work* / classical methods paragraph (after Verma et al., 2025), where the PDF claims a **Webster-method baseline** is included in the project                                                                                                                                   |
| **Old**   | *Paraphrase of current text:* “… **this project includes a Webster-method baseline alongside the fixed-time and actuated baselines**, allowing a direct comparison between a mathematically optimized static schedule and a data-driven adaptive policy under identical simulation conditions.” |
| **New**   | See block below                                                                                                                                                                                                                                                                                 |


**New:**

Webster’s formula (1958) remains a widely cited deterministic approach: given measured volumes and saturation flows, it computes an optimal cycle length and allocates green time across phases to reduce average delay. A recent four-arm intersection study reported substantial waiting reductions relative to an existing fixed plan (Verma et al., 2025). That method assumes relatively static demand and isolated operation, so it does not adapt minute-to-minute. **This project therefore uses reinforcement learning to learn a policy from repeated simulation, and it evaluates that policy against fixed-time and actuated controllers under shared simulation settings.** Webster’s method is retained here as **theoretical context** for why cycle-based optimization remains attractive—and why fully adaptive control may still be needed when demand is non-stationary.

*(If you later implement a Webster baseline in code, replace the last sentence with a sentence that points to its results in Section 7.)*

---

## 4. Section 3 (Concepts) — closing paragraph on final reward


|           |                                                                                                                    |
| --------- | ------------------------------------------------------------------------------------------------------------------ |
| **Where** | End of Section 3, after the **ThroughputQueueReward** description (currently ends with weight_1 / weight_2 tuning) |
| **Old**   | `empty` (no paragraph naming **ThroughputCompositeReward** as final) — *or delete nothing; this is an insertion.*  |
| **New**   | See block below                                                                                                    |


**New (insert after ThroughputQueueReward subsection):**

**Final deployed reward (reported system):** The configuration used for the final experiments and the Streamlit workflow is **ThroughputCompositeReward**, which combines throughput-oriented terms with queue and delay penalties using weights **α = 0.4**, **β = 0.15**, and **γ = 1.0** (with normalisation as configured in `reward_configuration.json`). Earlier reward designs (e.g., vehicle count deltas, **CompositeReward**, **ThroughputQueueReward**) were used during development and grid search to compare learning stability and crossing-rate behaviour; they are not the final reported reward unless explicitly stated in a table.

---

## 5. Section 4 — Objective O3


|           |                                                                                                                                                                                                                               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | *Project objectives* — bullet **O3**                                                                                                                                                                                          |
| **Old**   | *Paraphrase:* “O3. Experiment with multiple reward formulations — including wait-time, delta wait-time, throughput-only, composite, and throughput-queue — and select the most effective configuration for the final system.” |
| **New**   | See block below                                                                                                                                                                                                               |


**New:**

**O3.** Experiment with multiple reward formulations—including wait-time, delta wait-time, throughput-only, composite, throughput–queue, and **throughput–composite** variants—and document how the **final ThroughputCompositeReward** configuration was selected for the reported evaluation.

---

## 6. Section 4 — Objective O4


|           |                                                                                                                                                                                         |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Bullet **O4**                                                                                                                                                                           |
| **Old**   | *Paraphrase:* “O4. Evaluate the learned controller against fixed-time and actuated baseline policies using the same evaluation environment to enable a fair and meaningful comparison.” |
| **New**   | See block below                                                                                                                                                                         |


**New:**

**O4.** Evaluate the learned controller against fixed-time and actuated baselines using the **same simulator, network, timing constraints, and reward logging**, and report comparisons on **overlapping held-out test days** where the train/test split assigns different day IDs to the RL test set and the baseline batch (see Appendix B).

---

## 7. Section 5 — FR4.4


|           |                                                                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Table row **FR4.4** under Functional Requirements                                                                                  |
| **Old**   | *Paraphrase:* “The final reported system shall support the **Throughput-Queue** reward configuration for training and evaluation.” |
| **New**   | See block below                                                                                                                    |


**New:**

**FR4.4** — The final reported system shall support the **ThroughputCompositeReward** configuration for training and evaluation (resolved parameters recorded in the run directory).

---

## 8. Section 6 — output paths (`logs/` vs `src/data/results/`)


|           |                                                                                                          |
| --------- | -------------------------------------------------------------------------------------------------------- |
| **Where** | *Development Platform* / *Noteworthy Points* — sentence that checkpoints and logs live under `**logs/`** |
| **Old**   | *Paraphrase:* “… stored locally under **logs/** and related **src/data/** directories.”                  |
| **New**   | See block below                                                                                          |


**New:**

Checkpoints, trained models, and JSON logs are stored under `**src/data/results/<timestamp>_<policy>_<reward>/`** for CLI runs, and under the corresponding run directory when launched from Streamlit; grid-search outputs may appear under `**experiments/grid_search/runs/**`.

---

## 9. Section 7 — numbering (instruction only)


|           |                                                                                                                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Where** | Whole Section 7 structure                                                                                                                                                                                                |
| **Old**   | **7.X** Experimental Results (placeholder); Conclusions cite “Section 7.5” for grid search inconsistently                                                                                                                |
| **New**   | Suggested numbering: **7.4** System Validation (keep) → **7.5** Experimental Results → **7.6** Finding the right reward & hyperparameter tuning. Update **Table of Contents** and **all cross-references** in Section 9. |


---

## 10. Section 7.5 — Experimental Results (full subsection body)


|           |                                                                                                                                   |
| --------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Replace old **“7.X Experimental Results”** block (the paragraph starting with “The Double DQN agent was trained for 300 epochs…”) |
| **Old**   | Entire current **7.X Experimental Results** subsection (including the throughput sentence that said “per simulated hour window”)  |
| **New**   | See block below                                                                                                                   |


**New:**

**7.5 Experimental results**

The Double DQN agent was trained for **300 epochs** across **five** traffic days (**1,500** training episodes) using **ThroughputCompositeReward**, a cosine learning-rate schedule (**5×10⁻⁴ → 1×10⁻⁴**), and ε-greedy exploration from **1.0** toward **0.05**. At the end of training, **ε ≈ 0.492**, so exploration had **not** fully decayed and the policy should be interpreted as **partially converged** within the allotted budget.

On the **five** held-out test days used for the final agent, greedy evaluation produced a mean cumulative reward of **+1.61** per episode, compared with **−1,172.75** for the untrained random policy on the **same** five days (**+100.1%** improvement in mean reward).

For comparison to classical controllers, Table B1 (Appendix B) reports **mean reward on the three test days shared** by both the DQN evaluation and the fixed-time/actuated batch (**days 0, 1, and 4**): **−18.51** (DQN) versus **−1,551.19** (fixed-time) and **−1,698.20** (actuated)—approximately **98.8%** and **98.9%** higher mean reward on those days. DQN was also evaluated on two additional test days (**6** and **9**) that were **not** part of the baseline batch; those rows appear in Table B1 but are not averaged against baselines.

On the three overlapping days alone, mean reward improved from **−1838.64** (random policy) to **−18.51** (learned policy).

Across the five DQN test episodes, mean completed throughput was about **951 vehicles per 2-hour evaluation window**, with a mean **crossing rate** of about **13.2%** (as logged in `test_log.json`). The learned plan was exported as `**schedule.json`**.

---

## 11. Section 8 (Discussion) — “identical conditions” bullet


|           |                                                                                                                                                                                                                               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | *Impact of the System* — bullet ending with “pre-train and post-train … **identical conditions**”                                                                                                                             |
| **Old**   | *Paraphrase:* “The inclusion of a pre-train and post-train evaluation on held-out test days, alongside two baselines evaluated under **identical conditions**, produces results that are directly comparable and defensible…” |
| **New**   | See block below                                                                                                                                                                                                               |


**New:**

The evaluation uses the **same SUMO network, simulation window (28,800–36,000 s), reward definition, and controller constraints** for the learned agent and for baselines. **Test day IDs differ** between the final DQN test set and the baseline run archived for this report; therefore, headline comparisons use the **three overlapping test days** (0, 1, 4), while the **five-day pre-train versus post-train** comparison uses identical days for the RL runs only (Table B1, Appendix B).

---

## 12. Section 9 (Conclusions) — paragraph with +36.4% / +25.7%


|           |                                                                                                                                                                               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Second paragraph of Conclusions (after “The goal has been substantially achieved…”)                                                                                           |
| **Old**   | *Exact sense of old text:* Grid search (Section 7.5) showed the agent outperformed fixed-time by **+36.4%** and actuated by **+25.7%** under identical simulation conditions. |
| **New**   | See block below                                                                                                                                                               |


**New:**

Grid search over reward definitions and hyperparameters (**Section 7.6**) informed the choice of **ThroughputCompositeReward** and training settings before the final run. The primary empirical result, however, comes from the **final evaluation** (**Section 7.5**): on five held-out test days, mean cumulative reward improved from **−1,172.75** (random policy) to **+1.61** (greedy learned policy). On the **three** test days common to both the learned agent and the classical baselines, mean reward was **−18.51** versus **−1,551.19** (fixed-time) and **−1,698.20** (actuated)—approximately **98.8%** and **98.9%** higher mean reward on those shared days. Training had not fully converged (**ε ≈ 0.49** at 300 epochs), so further training or a slower exploration decay is expected to change absolute rewards but does not diminish the observed gap to baselines in this study.

*(If your grid-search section is numbered **7.5** and results **7.4**, swap the section numbers in this paragraph to match your TOC.)*

---

## 13. Appendix B — Table B1


|           |                                                                                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Appendix B table (partial / merged with Figure A1 in PDF — fix layout)                                                                                                                 |
| **Old**   | Table with only days 0, 1, 4 and a row “Mean (Days 0,1,4)” that put **−1172.75** in the pre-train column next to the three-day row (**incorrect** — −1172.75 is the **five-day** mean) |
| **New**   | See block below                                                                                                                                                                        |


**New — Table B1 + caption + footnote:**

**Table B1.** Cumulative reward by test day (2 h window: 28,800–36,000 s). DQN: greedy evaluation. Higher is better. Reward: **ThroughputCompositeReward** (α = 0.4, β = 0.15, γ = 1.0, normalised).


| Test day                  | DQN (post-train) | DQN (pre-train, random) | Fixed-time   | Actuated     |
| ------------------------- | ---------------- | ----------------------- | ------------ | ------------ |
| 0                         | +50.82           | −991.47                 | −1053.01     | −1209.91     |
| 1                         | +6.23            | −1408.79                | −1375.62     | −1405.78     |
| 4                         | −112.58          | −3115.65                | −2224.94     | −2478.92     |
| 6                         | +19.58           | +49.05                  | —            | —            |
| 9                         | +44.00           | −396.87                 | —            | —            |
| **Mean (all 5 DQN days)** | **+1.61**        | **−1172.75**            | —            | —            |
| **Mean (days 0, 1, 4)**   | **−18.51**       | **−1838.64**            | **−1551.19** | **−1698.20** |


**Footnote:** Baselines were evaluated on days **0, 1, 2, 4, 5**; the DQN run’s test split used days **0, 1, 4, 6, 9**. **Direct comparison** between controllers uses **days 0, 1, 4** only. Empty cells mean that baseline was not run for that test day.

---

## 14. “Finding the right reward” subsection


|           |                                                                                                                                                                                                                                                 |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where** | Fragment after Experimental Results (pages ~23–26 in BACKUP)                                                                                                                                                                                    |
| **Old**   | Long draft bullets + partial table (*no single clean “old” block* — tighten or move to appendix per page limit)                                                                                                                                 |
| **New**   | `empty` — **optional:** replace with one short summary paragraph + “Details in Appendix B / grid search logs” *or* keep your content but fix numbering to **Section 7.6** and ensure **ThroughputCompositeReward** is named as the final choice |


---

## Quick reference — numbers used (final run `2026-04-06_02-03-36_…`)


| Item                                        | Value    |
| ------------------------------------------- | -------- |
| Epochs                                      | 300      |
| Training episodes                           | 1,500    |
| ε at end                                    | ~0.492   |
| Mean reward, 5 DQN test days, post-train    | +1.61    |
| Mean reward, 5 DQN test days, pre-train     | −1172.75 |
| Mean reward, days 0+1+4, post-train         | −18.51   |
| Mean reward, days 0+1+4, pre-train          | −1838.64 |
| Mean reward, days 0+1+4, fixed-time         | −1551.19 |
| Mean reward, days 0+1+4, actuated           | −1698.20 |
| ~Vehicles per 2 h window (mean over 5 days) | ~951     |
| ~Mean crossing rate                         | ~13.2%   |


---

## 15. `schedule.json` as a **core feature** — **specific** placement (matches your report outline + actual UI)

**What the app actually shows (use these exact labels in the report):** After training, the **Results** subheader is followed by log tables and charts; if `schedule.json` exists, Streamlit shows **“Learned Signal Timing Schedule”** with a caption about **median phase durations (seconds) per 15-minute time bucket**, a **dataframe** (`tls_id`, `phase`, `time`, `median_s`, `std_s`, `n`), and `**Download schedule.json`**. The same block appears under **Load Existing Run Results** when a folder containing `schedule.json` is loaded. Your screenshot from `Report Screen shots/` should capture **that** block (heading + table + button).

---

### 1. Abstract


|                   |                                                                                                                                                                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | Last 1–2 sentences of the *Abstract*, **after** any mention of logs, checkpoints, or `schedule.json` as a file.                                                                                                                               |
| **Old**           | Abstract that only says the file exists **without** the **UI** (preview + download + load path).                                                                                                                                              |
| **New**           | Append: *“The Streamlit interface exposes this artefact under **Learned Signal Timing Schedule**, with a tabular preview and a **Download schedule.json** control, including when **loading an existing run directory** without retraining.”* |


---

### 2. Section 1 — *Introduction* (optional, one sentence)


|                   |                                                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | End of the **first** or **last** paragraph of Section 1 — **not** a new subsection.                                                                                                         |
| **Old**           | `empty` (if deliverables are not mentioned here).                                                                                                                                           |
| **New**           | *“In addition to model checkpoints, the system produces a **human-readable timing summary** (`schedule.json`) suitable for review and hand-off without inspecting neural network weights.”* |


---

### 3. Section 5 — *Table 5.1 Functional Requirements*


|                   |                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | **Insert a new row immediately after FR8.2** (generated plots). Use **FR8.3**.                                                                                                                                                                                                                                                                                                |
| **Old**           | `empty` row.                                                                                                                                                                                                                                                                                                                                                                  |
| **New**           | **FR8.3** — *The system shall write a `**schedule.json`** timing summary derived from evaluation (aggregated phase timing). The Streamlit front end shall display it when present (**Learned Signal Timing Schedule** with preview table) and shall provide **Download schedule.json**. The same behaviour shall apply when **loading** a completed run directory from disk.* |


---

### 4. Section 6 — *Development Strategy and Implementation*


|                   |                                                                                                                                                                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | In the **Streamlit** paragraph (or **Noteworthy Points**), **right after** the sentence about `streamlit run streamlit_app.py` / `streamlit_runs/` / “viewing streamed logs”.                                                                 |
| **Old**           | Paragraph that stops at logs/charts **without** schedule export.                                                                                                                                                                              |
| **New**           | Add: *“Runs invoke `**modelling/schedule_export.py`** to build `schedule.json`; `**streamlit_app.py**` surfaces it under **Learned Signal Timing Schedule** with **Download schedule.json**, including after **Load Existing Run Results**.”* |


---

### 5. Section 7.1.1 — *Component-Connector* — **External Services and Outputs layer**


|                   |                                                                                                                                                                         |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | In the **same sentence or bullet list** where you list `final_model.pt`, JSON logs, baseline logs, plots — **insert `schedule.json` in that list** (no new subsection). |
| **Old**           | List ending at “… generated plots to timestamped run directories.”                                                                                                      |
| **New**           | Add `**schedule.json`** *(aggregated learned phase timing per time bucket for export/inspection)*.                                                                      |


---

### 6. Section 7.3 — *Operational Evidence* — **Scenario B (Streamlit)** — **primary home for the screenshot**


|                   |                                                                                                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | **Inside Scenario B**, after the sentence that *Streamlit produces the same artefacts as CLI* and after **Figures 2–8** are referenced. **Then:** (1) paste the **new paragraph** below, (2) insert **Figure 9** (your screenshot). |


**New paragraph (paste after Scenario B narrative, before Scenario C — Grid Search):**

When training completes, the **Results** section lists train, pretest, and test logs and charts. If evaluation produced a timing export, the interface also shows **Learned Signal Timing Schedule**: a table of **median phase durations (seconds) per 15-minute bucket** (aggregated from test episodes), plus **Download schedule.json**. The same block appears when a user **loads an existing run** via **Load Existing Run Results**, so older runs remain reportable without re-simulation.

| **Figure number** | If BACKUP uses **Figures 2–8** for the UI workflow, add this as **Figure 9**. **Renumber** if you prefer schedule before baseline: keep one consistent story (e.g. train → **results/schedule** → baseline → plots). |
| **Caption (paste-ready)** | **Figure 9 — Streamlit Results: Learned Signal Timing Schedule and Download schedule.json.** After a completed training run (or after loading a run directory), the app shows median phase durations per 15-minute bucket and a download control for the full JSON timing plan. |
| **First in-text cite** | *“Figure 9 shows the **Learned Signal Timing Schedule** panel and **Download schedule.json** button.”* |

---

### 7. Section 7.5 — *Experimental results*


|                   |                                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | **Last paragraph** of the subsection, **in the same sentence** that mentions `schedule.json` export.                                                      |
| **Old**           | *“… exported as `schedule.json`.”* (sentence end)                                                                                                         |
| **New**           | *“… exported as `**schedule.json*`*; the Streamlit **Results** view (Figure 9) provides a tabular preview and **Download schedule.json** for reporting.”* |


---

### 8. Section 8 — *Discussion* — *Impact of the System* (optional)


|                   |                                                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Where exactly** | **New bullet** under *Impact*, after the Streamlit accessibility bullet **or** after modular design.                                                                                       |
| **Old**           | `empty`                                                                                                                                                                                    |
| **New**           | *“The `**schedule.json`** export (with UI preview and download) turns the learned policy into a **reviewable timing artefact** for stakeholders who do not use PyTorch or SUMO directly.”* |


---

### 9. Section 9 — *Conclusions* — technical artefacts paragraph


|                   |                                                                                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | Paragraph listing **Streamlit** (§7.3), **schedule export module**, **ThroughputCompositeReward** — **extend** the clause on the schedule export module.                                              |
| **Old**           | *“… schedule export module that distils the learned policy into a deployable fixed-time signal plan.”*                                                                                                |
| **New**           | *“… schedule export module that distils the learned policy into a deployable fixed-time signal plan, **exposed in the Streamlit UI as Learned Signal Timing Schedule with Download schedule.json**.”* |


---

### 10. List of Figures (if required)


|                   |                                                             |
| ----------------- | ----------------------------------------------------------- |
| **Where exactly** | List of Figures / auto TOC — **one new line** for Figure 9. |


---

### 11. Appendix


|                   |                                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Where exactly** | **Do not** put the screenshot here **unless** you must move §7.3 figures for page limits. **Avoid** duplicating the same figure in §7.3 and Appendix. |


---

### Summary checklist


| §        | Subsection           | Action                                                                                                          |
| -------- | -------------------- | --------------------------------------------------------------------------------------------------------------- |
| Abstract | —                    | +1 sentence (UI: **Learned Signal Timing Schedule**, **Download schedule.json**, **Load Existing Run Results**) |
| 1        | Introduction         | optional +1 sentence                                                                                            |
| 5        | FR table             | new **FR8.3** after FR8.2                                                                                       |
| 6        | Streamlit            | +1 sentence (`schedule_export`, `streamlit_app.py`)                                                             |
| 7.1.1    | Outputs list         | add `**schedule.json`** to artefact list                                                                        |
| **7.3**  | **Scenario B**       | **+paragraph + Figure 9 (screenshot from Report Screen shots)**                                                 |
| 7.5      | Experimental results | +clause + cross-ref Figure 9                                                                                    |
| 8        | Impact               | optional bullet                                                                                                 |
| 9        | Conclusions          | extend artefacts sentence                                                                                       |


---

*Generated for RL-Traffic-Light-Optimization report sync. Adjust section numbers (7.4 / 7.5 / 7.6) to match your final outline.*