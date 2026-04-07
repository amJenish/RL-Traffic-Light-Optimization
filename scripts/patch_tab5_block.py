"""One-off splice: replace Tab 5 block in streamlit_app.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "streamlit_app.py"
text = path.read_text(encoding="utf-8")
start = text.index("    with tabs[4]:")
end = text.index("\n\nif __name__ == \"__main__\":", start)

NEW = r'''    with tabs[4]:
        st.divider()
        st.subheader("Already have a timing plan? Compare it against the mathematical baseline.")
        if "tab5_results_mode" not in st.session_state:
            st.session_state.tab5_results_mode = (
                TAB5_LABEL_TRAINING if st.session_state.run_complete else TAB5_LABEL_UPLOAD
            )
        st.radio(
            "Results view",
            [TAB5_LABEL_TRAINING, TAB5_LABEL_UPLOAD],
            key="tab5_results_mode",
            label_visibility="collapsed",
        )
        if st.session_state.tab5_results_mode == TAB5_LABEL_TRAINING:
            st.session_state.tab5_mode = "training"
            _render_tab5_training_results()
        else:
            st.session_state.tab5_mode = "upload"
            S2_USE_DATA = "Use the file I already uploaded in the Data tab"
            S2_UPLOAD = "Upload a different traffic count file"
            S3_USE_INTER = "Use the intersection I already configured in the Intersection tab"
            S3_UPLOAD = "Upload configuration files"
            S4_REGEN = "Re-generate from uploaded data (recommended if no previous run)"
            S4_PREV = "Use files from a previous run directory"

            st.markdown("### Step 1: Upload your timing plan")
            st.file_uploader(
                "Upload your signal timing plan",
                type=["json"],
                key="upload_schedule_json",
                help="The timing plan file from a previous training run (expects tls_id J_centre in buckets).",
            )
            up_sched = st.session_state.get("upload_schedule_json")
            step1_ok = False
            step1_line = "Step 1: ○ Waiting for file"
            if up_sched is not None:
                try:
                    raw = json.loads(up_sched.getvalue().decode("utf-8"))
                    if _valid_upload_schedule_dict(raw):
                        st.session_state.upload_schedule_dict = raw
                        nb = len(raw["buckets"])
                        ph = len({int(b["phase"]) for b in raw["buckets"]})
                        step1_ok = True
                        step1_line = f"Step 1: ✓ Timing plan loaded — {nb} buckets across {ph} phases."
                        st.success(step1_line.replace("Step 1: ✓ ", "✓ "))
                    else:
                        st.session_state.upload_schedule_dict = None
                        st.warning(
                            "✗ This file does not look like a valid timing plan. Please upload the correct file."
                        )
                        step1_line = "Step 1: ✗ Invalid timing plan file"
                except Exception:
                    st.session_state.upload_schedule_dict = None
                    st.warning(
                        "✗ This file does not look like a valid timing plan. Please upload the correct file."
                    )
                    step1_line = "Step 1: ✗ Invalid timing plan file"
            else:
                st.session_state.upload_schedule_dict = None

            st.markdown("### Step 2: Provide traffic data")
            st.radio("Traffic data source", [S2_USE_DATA, S2_UPLOAD], key="upload_step2_traffic_source")
            step2_ok = False
            step2_line = "Step 2: ○ Waiting for file"
            if st.session_state.upload_step2_traffic_source == S2_USE_DATA:
                if st.session_state.csv_df is not None:
                    step2_ok = True
                    step2_line = "Step 2: ✓ Using previously uploaded traffic data."
                    st.success("✓ Using previously uploaded traffic data.")
                else:
                    st.warning(
                        "⚠ No file found in the Data tab. Please upload one there first, or choose to upload a different file here."
                    )
            else:
                cf = st.file_uploader(
                    "Upload a traffic count file",
                    type=["csv", "xlsx"],
                    key="upload_compare_csv",
                    help="A traffic count file covering the time period you want to evaluate. Used to compute the mathematical baseline.",
                )
                if cf is not None:
                    try:
                        if cf.name.lower().endswith(".xlsx"):
                            tdf = pd.read_excel(io.BytesIO(cf.getvalue()))
                        else:
                            tdf = pd.read_csv(io.BytesIO(cf.getvalue()))
                        num_any = any(
                            pd.api.types.is_numeric_dtype(tdf[c]) for c in tdf.columns
                        )
                        if num_any and len(tdf) > 0:
                            st.session_state.upload_compare_csv_df = tdf
                            step2_ok = True
                            step2_line = f"Step 2: ✓ Traffic file loaded — {len(tdf)} rows."
                            st.success(f"✓ Traffic file loaded — {len(tdf)} rows.")
                            st.dataframe(tdf.head(3), use_container_width=True)
                        else:
                            st.session_state.upload_compare_csv_df = None
                            st.warning("✗ Could not validate traffic file (need at least one numeric column and rows).")
                    except Exception:
                        st.session_state.upload_compare_csv_df = None
                        st.warning("✗ Could not read traffic file.")
                else:
                    st.session_state.upload_compare_csv_df = None

            st.markdown("### Step 3: Provide intersection and column mapping")
            st.radio(
                "Intersection configuration source",
                [S3_USE_INTER, S3_UPLOAD],
                key="upload_step3_config_source",
            )
            step3_ok = False
            step3_line = "Step 3: ○ Waiting"
            inter_use: dict | None = None
            col_use: dict | None = None
            if st.session_state.upload_step3_config_source == S3_USE_INTER:
                if st.session_state.intersection_text and st.session_state.columns_text:
                    try:
                        inter_use = json.loads(st.session_state.intersection_text)
                        col_use = json.loads(st.session_state.columns_text)
                        if _valid_intersection_dict_upload(inter_use) and _valid_columns_dict_upload(col_use):
                            step3_ok = True
                            nm = inter_use.get("intersection_name", "Intersection")
                            step3_line = f"Step 3: ✓ Using intersection: {nm}"
                            st.success(f"✓ Using intersection: {nm}")
                        else:
                            st.warning("⚠ Invalid intersection or columns JSON in session.")
                    except Exception:
                        st.warning("⚠ Could not parse intersection or columns from the Intersection tab.")
                else:
                    st.warning(
                        "⚠ No intersection configured. Set one up in the Intersection tab, or upload configuration files here."
                    )
            else:
                cL, cR = st.columns(2)
                with cL:
                    st.file_uploader(
                        "Intersection configuration",
                        type=["json"],
                        key="upload_compare_intersection",
                        help="The intersection.json file from your previous run.",
                    )
                with cR:
                    st.file_uploader(
                        "Column mapping",
                        type=["json"],
                        key="upload_compare_columns",
                        help="The columns.json file that maps your CSV column headers.",
                    )
                fi = st.session_state.get("upload_compare_intersection")
                fc = st.session_state.get("upload_compare_columns")
                if fi is not None and fc is not None:
                    try:
                        inter_use = json.loads(fi.getvalue().decode("utf-8"))
                        col_use = json.loads(fc.getvalue().decode("utf-8"))
                        if _valid_intersection_dict_upload(inter_use):
                            st.success(f"✓ Intersection: {inter_use.get('intersection_name', '')}")
                        else:
                            inter_use = None
                            st.warning("✗ Invalid intersection file.")
                        if _valid_columns_dict_upload(col_use):
                            st.success("✓ Column mapping loaded.")
                        else:
                            col_use = None
                            st.warning("✗ Invalid column mapping file.")
                        step3_ok = bool(inter_use and col_use)
                        if step3_ok:
                            step3_line = f"Step 3: ✓ Intersection: {inter_use.get('intersection_name', '')}"
                    except Exception:
                        inter_use, col_use = None, None
                        st.warning("✗ Could not parse uploaded configuration files.")

            st.markdown("### Step 4: Network and simulation files")
            st.radio("Network and flows", [S4_REGEN, S4_PREV], key="upload_step4_network_mode")
            step4_ok = False
            step4_line = "Step 4: ○ Will be generated"
            n_eval_note = 0
            if st.session_state.upload_step4_network_mode == S4_REGEN:
                st.info(
                    "The road network and demand files will be built automatically from your traffic data and "
                    "intersection configuration when you run the comparison. This takes about the same time as "
                    "the network-building step in a full training run."
                )
                st.number_input(
                    "How many days should be used for evaluation?",
                    min_value=1,
                    max_value=30,
                    value=5,
                    key="upload_compare_test_days",
                    help="The comparison will be run across this many simulated test days.",
                )
                step4_ok = True
                step4_line = "Step 4: ✓ Network files ready (will be generated)"
            else:
                st.text_input(
                    "Path to previous run folder",
                    key="upload_compare_run_dir",
                    placeholder="e.g. src/data/results/2024-01-15_10-30-00_...",
                    help="Folder containing processed/ with split.json and sumo/ with network and flows.",
                )
                prd = str(st.session_state.get("upload_compare_run_dir") or "").strip()
                inm = ""
                if inter_use:
                    inm = str(inter_use.get("intersection_name", ""))
                if prd and inm:
                    okp, msgp, n_ev = _validate_prev_run_dir(prd, inm)
                    st.caption(msgp)
                    step4_ok = okp
                    n_eval_note = n_ev
                    if okp:
                        step4_line = f"Step 4: ✓ Network files ready ({n_ev} test days)"
                elif prd and not inm:
                    st.caption("⚠ Set intersection (Step 3) before validating the run folder.")
                else:
                    st.caption("⚠ Enter a path to your previous run folder.")

            if st.session_state.upload_step4_network_mode == S4_PREV and step4_ok:
                st.info(f"Using {n_eval_note} test days from the selected run.")

            st.markdown("---")
            st.markdown("**Readiness**")
            st.caption(step1_line)
            st.caption(step2_line)
            st.caption(step3_line)
            st.caption(step4_line)

            ready = step1_ok and step2_ok and step3_ok and step4_ok
            if st.button(
                "▶  Run comparison",
                type="primary",
                key="run_upload_compare",
                disabled=not ready or st.session_state.upload_compare_in_progress,
            ):
                st.session_state.upload_compare_in_progress = True
                st.session_state.upload_compare_complete = False
                st.session_state.upload_compare_error = ""
                st.session_state.upload_compare_log_lines = []
                st.session_state.upload_compare_stage_statuses = ["waiting"] * 4
                st.session_state.upload_compare_results = None
                try:
                    if st.session_state.upload_step2_traffic_source == S2_USE_DATA:
                        tdf_run = st.session_state.csv_df.copy()
                    else:
                        tdf_run = st.session_state.upload_compare_csv_df.copy()
                    assert inter_use is not None and col_use is not None
                    assert st.session_state.upload_schedule_dict is not None
                    tr_run = int(st.session_state.get("ui_train_days", train_cfg.get("train_days", 5)))
                    if st.session_state.upload_step4_network_mode == S4_REGEN:
                        td_run = int(st.session_state.get("upload_compare_test_days", 5))
                    else:
                        pr_run = str(st.session_state.get("upload_compare_run_dir") or "").strip()
                        in_run = str(inter_use.get("intersection_name", "")) if inter_use else ""
                        _ok_r, _msg_r, n_ev_r = _validate_prev_run_dir(pr_run, in_run)
                        td_run = int(n_ev_r) if _ok_r else int(st.session_state.get("ui_test_days", 5))

                    ui_uc = {
                        "n_epochs": int(st.session_state.get("ui_epochs", train_cfg.get("n_epochs", 200))),
                        "train_days": tr_run,
                        "test_days": td_run,
                        "decision_gap": int(st.session_state.get("ui_decision_gap", sim_cfg.get("decision_gap", 10))),
                        "seed": int(st.session_state.get("ui_seed", train_cfg.get("seed", 42))),
                        "learning_rate": float(st.session_state.get("ui_learning_rate", policy_lr_default)),
                        "lr_min": float(st.session_state.get("ui_lr_min", train_cfg.get("lr_min", 0.0001))),
                        "step_length": float(st.session_state.get("ui_step_length", sim_cfg.get("step_length", 1.0))),
                        "sim_begin": int(_time_to_seconds(st.session_state.ui_sim_start)),
                        "sim_end": int(_time_to_seconds(st.session_state.ui_sim_end)),
                        "subslots_per_slot": int(st.session_state.get("ui_subslots", flow_cfg.get("subslots_per_slot", 5))),
                        "spread": float(st.session_state.get("ui_spread", flow_cfg.get("spread", 0.85))),
                        "reward_class": str(st.session_state.get("ui_reward_class", default_reward)),
                        "sumo_home": st.session_state.get("ui_sumo_home", os.environ.get("SUMO_HOME", "")),
                        "run_baseline_comparison": False,
                    }

                    def _pcb(msg: str) -> None:
                        st.session_state.upload_compare_log_lines.append(str(msg))

                    def _su(i: int, s: str) -> None:
                        st.session_state.upload_compare_stage_statuses[i] = s

                    res = _run_upload_compare_pipeline(
                        _pcb,
                        _su,
                        base_cfg=base_cfg,
                        ui=ui_uc,
                        compare_root=COMPARE_STANDALONE,
                        schedule_dict=st.session_state.upload_schedule_dict,
                        traffic_df=tdf_run,
                        intersection_dict=inter_use,
                        columns_dict=col_use,
                        regenerate_network=st.session_state.upload_step4_network_mode == S4_REGEN,
                        prev_run_dir=str(st.session_state.get("upload_compare_run_dir") or "").strip(),
                        sumo_home=str(ui_uc["sumo_home"]),
                        train_days=tr_run,
                        test_days=td_run,
                    )
                    if res["ok"]:
                        st.session_state.upload_compare_complete = True
                        st.session_state.upload_compare_results = res
                    else:
                        st.session_state.upload_compare_error = res.get("error") or "Comparison failed."
                except Exception as ex:
                    st.session_state.upload_compare_error = traceback.format_exc()
                finally:
                    st.session_state.upload_compare_in_progress = False

            if st.session_state.get("upload_compare_error"):
                st.error("Something went wrong during the upload comparison pipeline.")
                st.code(st.session_state.upload_compare_error, language="text")

            st.markdown("### Progress")
            ust = st.session_state.get("upload_compare_stage_statuses", ["waiting"] * 4)
            ulab = [
                "Building demand and network files",
                "Computing mathematical baseline timing plan",
                "Running both timing plans through simulation",
                "Comparing results",
            ]
            uicon = {"waiting": "○", "active": "⏳", "done": "✓", "failed": "✗", "skipped": "–"}
            for i, lab in enumerate(ulab):
                st.write(f"{uicon.get(ust[i], '○')} {i + 1}. {lab}")

            with st.expander("Detailed log", expanded=False):
                st.text_area(
                    "Log output",
                    value="\n".join(st.session_state.upload_compare_log_lines[-800:]),
                    height=280,
                    key="upload_compare_log_view",
                )

            if st.session_state.upload_compare_complete and st.session_state.upload_compare_results:
                ur = st.session_state.upload_compare_results
                if ur.get("ok") and ur.get("paths"):
                    _render_upload_compare_result_section(
                        ur["paths"],
                        int(ur.get("n_test_days", 0)),
                    )
'''

path.write_text(text[:start] + NEW + text[end:], encoding="utf-8")
print("patched", path)
