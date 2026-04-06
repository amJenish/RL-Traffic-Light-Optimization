"""
Friendly editors for columns.json + intersection.json — field-based UI synced to session JSON strings.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import streamlit as st

from streamlit_intersection_helpers import DEFAULT_TIMING, _slug_name

DIRS = ["N", "S", "E", "W"]

_DEFAULT_COL_INSTRUCTIONS = (
    "Map CSV column names to the standard names used by this system. "
    "Leave a field empty if that column does not exist."
)
_DEFAULT_COL_NOTES = [
    "If data has no right-turn or left-turn columns, leave those empty — they are treated as absent.",
    "If data has no separate date column, leave date empty.",
    "slot_minutes must match the interval between rows (e.g. 15 for 15-min counts).",
    "Approaches not in active_approaches are ignored even if mapped here.",
]


def _text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode()).hexdigest()


def invalidate_config_widget_sync() -> None:
    st.session_state.pop("_cm_widgets_hash", None)
    st.session_state.pop("_int_widgets_hash", None)


def _preserved_columns_meta() -> dict[str, Any]:
    raw = st.session_state.get("columns_text", "")
    if not raw.strip():
        return {"_instructions": _DEFAULT_COL_INSTRUCTIONS, "_notes": _DEFAULT_COL_NOTES}
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {"_instructions": _DEFAULT_COL_INSTRUCTIONS, "_notes": _DEFAULT_COL_NOTES}
    notes = d.get("_notes")
    if not isinstance(notes, list):
        notes = _DEFAULT_COL_NOTES
    return {
        "_instructions": d.get("_instructions") or _DEFAULT_COL_INSTRUCTIONS,
        "_notes": notes,
    }


def _field_or_null(key: str) -> str | None:
    v = st.session_state.get(key, "")
    if isinstance(v, str):
        v = v.strip()
    return v if v else None


def _sync_column_widgets_from_json() -> None:
    raw = st.session_state.get("columns_text", "")
    h = _text_hash(raw)
    if st.session_state.get("_cm_widgets_hash") == h:
        return
    st.session_state["_cm_widgets_hash"] = h
    if not str(raw).strip():
        st.session_state["ui_cm_time_start"] = ""
        st.session_state["ui_cm_time_end"] = ""
        st.session_state["ui_cm_time_date"] = ""
        st.session_state["ui_cm_time_dow"] = ""
        st.session_state["ui_cm_slot"] = 15
        for d in DIRS:
            for k in ("through", "right", "left", "peds"):
                st.session_state[f"ui_cm_{d}_{k}"] = ""
        return
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return
    t = d.get("time") or {}

    def _tget(x: str) -> str:
        v = t.get(x)
        return "" if v is None else str(v)

    st.session_state["ui_cm_time_start"] = _tget("start_time")
    st.session_state["ui_cm_time_end"] = _tget("end_time")
    st.session_state["ui_cm_time_date"] = _tget("date")
    st.session_state["ui_cm_time_dow"] = _tget("day_of_week")
    st.session_state["ui_cm_slot"] = int(d.get("slot_minutes") or 15)
    approaches = d.get("approaches") or {}
    for ddir in DIRS:
        block = approaches.get(ddir) or {}
        for k in ("through", "right", "left", "peds"):
            v = block.get(k)
            st.session_state[f"ui_cm_{ddir}_{k}"] = "" if v is None else str(v)


def _flush_columns_text_from_widgets() -> None:
    if not st.session_state.get("_had_config_once"):
        return
    meta = _preserved_columns_meta()
    time_block: dict[str, Any] = {
        "start_time": _field_or_null("ui_cm_time_start"),
        "end_time": _field_or_null("ui_cm_time_end"),
        "date": _field_or_null("ui_cm_time_date"),
        "day_of_week": _field_or_null("ui_cm_time_dow"),
    }
    approaches: dict[str, Any] = {}
    for ddir in DIRS:
        approaches[ddir] = {
            "through": _field_or_null(f"ui_cm_{ddir}_through"),
            "right": _field_or_null(f"ui_cm_{ddir}_right"),
            "left": _field_or_null(f"ui_cm_{ddir}_left"),
            "peds": _field_or_null(f"ui_cm_{ddir}_peds"),
        }
    slot = int(st.session_state.get("ui_cm_slot") or 15)
    out: dict[str, Any] = {
        "_instructions": meta["_instructions"],
        "time": time_block,
        "approaches": approaches,
        "slot_minutes": slot,
    }
    if meta.get("_notes"):
        out["_notes"] = meta["_notes"]
    st.session_state["columns_text"] = json.dumps(out, indent=2)


def _sync_intersection_widgets_from_json() -> None:
    raw = st.session_state.get("intersection_text", "")
    h = _text_hash(raw)
    if st.session_state.get("_int_widgets_hash") == h:
        return
    st.session_state["_int_widgets_hash"] = h
    if not str(raw).strip():
        st.session_state["ui_int_name"] = "My_Intersection"
        st.session_state["ui_int_active"] = ["N", "S", "E", "W"]
        for ddir in DIRS:
            st.session_state[f"ui_int_{ddir}_lt"] = 2
            st.session_state[f"ui_int_{ddir}_lr"] = 1
            st.session_state[f"ui_int_{ddir}_ll"] = 1
            st.session_state[f"ui_int_{ddir}_sp"] = 50
        st.session_state["ui_int_ph"] = str(DEFAULT_TIMING["phases"])
        st.session_state["ui_int_mr"] = int(DEFAULT_TIMING["min_red_s"])
        st.session_state["ui_int_mg"] = int(DEFAULT_TIMING["min_green_s"])
        st.session_state["ui_int_mx"] = int(DEFAULT_TIMING["max_green_s"])
        st.session_state["ui_int_am"] = int(DEFAULT_TIMING["amber_s"])
        st.session_state["ui_int_cy"] = int(DEFAULT_TIMING["cycle_s"])
        st.session_state["ui_int_el"] = int(DEFAULT_TIMING["edge_length_m"])
        return
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return
    st.session_state["ui_int_name"] = str(d.get("intersection_name") or "My_Intersection")
    act = d.get("active_approaches") or ["N", "S", "E", "W"]
    st.session_state["ui_int_active"] = [x for x in act if x in DIRS] or ["N", "S", "E", "W"]
    ap = d.get("approaches") or {}
    for ddir in DIRS:
        blk = ap.get(ddir) or {}
        lanes = blk.get("lanes") or {}
        st.session_state[f"ui_int_{ddir}_lt"] = int(lanes.get("through", 2))
        st.session_state[f"ui_int_{ddir}_lr"] = int(lanes.get("right", 1))
        st.session_state[f"ui_int_{ddir}_ll"] = int(lanes.get("left", 1))
        st.session_state[f"ui_int_{ddir}_sp"] = int(float(blk.get("speed_kmh", 50)))
    st.session_state["ui_int_ph"] = str(d.get("phases", DEFAULT_TIMING["phases"]))
    st.session_state["ui_int_mr"] = int(d.get("min_red_s", DEFAULT_TIMING["min_red_s"]))
    st.session_state["ui_int_mg"] = int(d.get("min_green_s", DEFAULT_TIMING["min_green_s"]))
    st.session_state["ui_int_mx"] = int(d.get("max_green_s", DEFAULT_TIMING["max_green_s"]))
    st.session_state["ui_int_am"] = int(d.get("amber_s", DEFAULT_TIMING["amber_s"]))
    st.session_state["ui_int_cy"] = int(d.get("cycle_s", DEFAULT_TIMING["cycle_s"]))
    st.session_state["ui_int_el"] = int(d.get("edge_length_m", DEFAULT_TIMING["edge_length_m"]))


def _flush_intersection_text_from_widgets() -> None:
    if not st.session_state.get("_had_config_once"):
        return
    name = str(st.session_state.get("ui_int_name") or "My_Intersection").strip() or "My_Intersection"
    active = st.session_state.get("ui_int_active") or []
    active = [a for a in active if a in DIRS]
    if not active:
        active = ["N", "S", "E", "W"]
    approaches: dict[str, Any] = {}
    for ddir in active:
        approaches[ddir] = {
            "lanes": {
                "through": int(st.session_state.get(f"ui_int_{ddir}_lt", 2)),
                "right": int(st.session_state.get(f"ui_int_{ddir}_lr", 1)),
                "left": int(st.session_state.get(f"ui_int_{ddir}_ll", 1)),
            },
            "speed_kmh": float(st.session_state.get(f"ui_int_{ddir}_sp", 50)),
        }
    out = {
        "intersection_name": _slug_name(name),
        "active_approaches": active,
        "approaches": approaches,
        "phases": str(st.session_state.get("ui_int_ph", "4")),
        "min_red_s": int(st.session_state.get("ui_int_mr", DEFAULT_TIMING["min_red_s"])),
        "min_green_s": int(st.session_state.get("ui_int_mg", DEFAULT_TIMING["min_green_s"])),
        "max_green_s": int(st.session_state.get("ui_int_mx", DEFAULT_TIMING["max_green_s"])),
        "amber_s": int(st.session_state.get("ui_int_am", DEFAULT_TIMING["amber_s"])),
        "cycle_s": int(st.session_state.get("ui_int_cy", DEFAULT_TIMING["cycle_s"])),
        "edge_length_m": int(st.session_state.get("ui_int_el", DEFAULT_TIMING["edge_length_m"])),
    }
    st.session_state["intersection_text"] = json.dumps(out, indent=2)


def render_column_and_intersection_forms() -> None:
    """Renders field editors and keeps columns_text / intersection_text in session_state in sync."""
    st.caption("Open **2** and **3** below to edit the column map and intersection. **Generate from CSV** fills both.")
    with st.expander("2 — Column map (your CSV columns)", expanded=False):
        st.caption(
            "Match each traffic movement to the **exact column name** in your CSV. Leave a box empty if you do not "
            "have that column."
        )
        _sync_column_widgets_from_json()

        st.markdown("**Time columns**")
        ct1, ct2, ct3, ct4 = st.columns(4)
        with ct1:
            st.text_input("Start time column", key="ui_cm_time_start", placeholder="e.g. start_time")
        with ct2:
            st.text_input("End time column", key="ui_cm_time_end", placeholder="e.g. end_time")
        with ct3:
            st.text_input("Date column", key="ui_cm_time_date", placeholder="e.g. date")
        with ct4:
            st.text_input("Day of week column", key="ui_cm_time_dow", placeholder="e.g. day_of_week")

        st.number_input(
            "Minutes per row in your CSV (slot length)",
            min_value=1,
            max_value=1440,
            step=1,
            key="ui_cm_slot",
            help="e.g. 15 for 15-minute counts, 60 for hourly.",
        )

        for ddir in DIRS:
            st.markdown(f"**Approach {ddir}** — through / right / left / pedestrians")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.text_input(
                    "Through",
                    key=f"ui_cm_{ddir}_through",
                    placeholder=f"{ddir.lower()}_approaching_t",
                )
            with a2:
                st.text_input(
                    "Right turn",
                    key=f"ui_cm_{ddir}_right",
                    placeholder=f"{ddir.lower()}_approaching_r",
                )
            with a3:
                st.text_input(
                    "Left turn",
                    key=f"ui_cm_{ddir}_left",
                    placeholder=f"{ddir.lower()}_approaching_l",
                )
            with a4:
                st.text_input("Pedestrians (optional)", key=f"ui_cm_{ddir}_peds", placeholder="optional")

        _flush_columns_text_from_widgets()

    with st.expander("3 — Intersection (lanes, speed, signal timing)", expanded=False):
        _sync_intersection_widgets_from_json()

        st.text_input("Intersection name (used in SUMO file names)", key="ui_int_name")
        st.multiselect(
            "Active approaches (directions with traffic)",
            options=DIRS,
            key="ui_int_active",
        )

        st.markdown("**Lanes and speed per direction**")
        for ddir in DIRS:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
            with c1:
                st.number_input(
                    f"{ddir} — through lanes",
                    min_value=1,
                    max_value=8,
                    key=f"ui_int_{ddir}_lt",
                )
            with c2:
                st.number_input(
                    f"{ddir} — right",
                    min_value=0,
                    max_value=8,
                    key=f"ui_int_{ddir}_lr",
                )
            with c3:
                st.number_input(
                    f"{ddir} — left",
                    min_value=0,
                    max_value=8,
                    key=f"ui_int_{ddir}_ll",
                )
            with c4:
                st.number_input(
                    f"{ddir} — speed (km/h)",
                    min_value=20,
                    max_value=130,
                    key=f"ui_int_{ddir}_sp",
                )

        st.markdown("**Signal timing (seconds & geometry)**")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.text_input("Phases (string, e.g. 4)", key="ui_int_ph")
            st.number_input("Min red (s)", min_value=1, max_value=120, key="ui_int_mr")
        with t2:
            st.number_input("Min green (s)", min_value=1, max_value=300, key="ui_int_mg")
            st.number_input("Max green (s)", min_value=1, max_value=600, key="ui_int_mx")
        with t3:
            st.number_input("Amber (s)", min_value=1, max_value=30, key="ui_int_am")
            st.number_input("Cycle (s)", min_value=30, max_value=300, key="ui_int_cy")
            st.number_input("Edge length (m)", min_value=10, max_value=500, key="ui_int_el")

        _flush_intersection_text_from_widgets()
