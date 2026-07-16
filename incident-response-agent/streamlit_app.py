"""
streamlit_app.py
-----------------
Optional local UI for exploring the incident-response agent interactively,
instead of reading CLI output. Not deployed anywhere (this environment has
no network/hosting access) — run it locally:

    pip install streamlit
    streamlit run streamlit_app.py

This is intentionally a thin presentation layer over the existing graph —
it imports `graph.py`/`app.py` directly rather than re-implementing any
logic, so there is exactly one code path for running an incident, whether
you invoke it from the CLI or this UI.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

import streamlit as st

from app import SAMPLE_ALERTS
from evaluation.trajectory_eval import score_trajectory
from graph import build_graph
from state import new_incident_state

st.set_page_config(page_title="Incident Response Agent", layout="wide")

st.title("Autonomous IT Incident Response — Live Run")
st.caption(
    "Runs the real LangGraph workflow in this repo (graph.py) against a mock tool/LLM layer. "
    "No incidents are simulated here that aren't also reachable via `python app.py`."
)

scenario = st.selectbox(
    "Choose a scenario",
    options=list(SAMPLE_ALERTS.keys()),
    format_func=lambda k: {
        "checkout": "checkout — dependency degradation \u2192 escalate",
        "payments": "payments — OOM \u2192 scale_service",
        "unknown": "notification — generic error \u2192 restart_service (with retry demo)",
        "transient": "reporting — no evidence \u2192 low-confidence escalate",
    }.get(k, k),
)

if st.button("Run incident", type="primary"):
    alert = SAMPLE_ALERTS[scenario]
    with st.spinner("Running graph..."):
        app = build_graph()
        final_state = app.invoke(
            new_incident_state(alert), config={"configurable": {"thread_id": str(uuid.uuid4())}}
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Incident Report")
        st.markdown(final_state["final_report"])

        st.subheader("Trajectory")
        for i, d in enumerate(final_state["decisions"], 1):
            st.markdown(f"**{i}. [{d.agent}]** {d.decision}")
            st.caption(d.reasoning)

        st.subheader("Tool Calls")
        for tc in final_state["tool_calls"]:
            icon = "\u2705" if tc.success else "\u274c"
            st.markdown(f"{icon} `{tc.tool_name}({tc.arguments})` — {tc.latency_ms:.0f}ms")
            if not tc.success:
                st.caption(f"error: {tc.error}")

    with col2:
        st.subheader("Trajectory Score")
        trajectory = dict(final_state)
        trajectory["tool_calls"] = [asdict(tc) for tc in final_state["tool_calls"]]
        trajectory["decisions"] = [asdict(d) for d in final_state["decisions"]]
        score = score_trajectory(trajectory).as_dict()

        st.metric("Task success", "Yes" if score["task_success"] else "No")
        st.metric("Retries", final_state["retry_count"])
        st.metric("Tool calls", score["tool_call_count"])
        st.metric("Total latency (ms)", f"{score['total_latency_ms']:.0f}")
        for k in ["planning_quality", "tool_selection_accuracy", "recovery_from_failure", "hallucinated_tool_rate", "state_consistency"]:
            st.progress(min(1.0, score[k]), text=f"{k}: {score[k]:.2f}")
        if score["notes"]:
            st.subheader("Evaluator notes")
            for n in score["notes"]:
                st.caption(f"\u2022 {n}")
