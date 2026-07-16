"""
graph.py
--------
Wires the agent nodes (agents.py) into a LangGraph `StateGraph`.

Topology
--------
    START -> monitor -> diagnosis -> decision -> execution -> verification -> report -> END
                                                                    |
                                                                    v (not resolved, retries remain)
                                                              decision  (loop back)

`verification` conditionally routes back to `decision` if the incident is
not yet resolved and the retry budget isn't exhausted, otherwise it moves
to `report`. This conditional edge is what turns a linear pipeline into
an actual agentic loop, and it's the piece a pure prompt-chaining
approach (no graph) cannot express cleanly.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents import (
    decision_agent,
    diagnosis_agent,
    execution_agent,
    monitor_agent,
    report_agent,
    verification_agent,
)
from state import IncidentState, IncidentStatus

MAX_REMEDIATION_LOOPS = 2


def _route_after_verification(state: IncidentState) -> str:
    if state["status"] == IncidentStatus.RESOLVED.value:
        return "report"
    if state["status"] == IncidentStatus.ESCALATED.value:
        return "report"
    if state["retry_count"] >= MAX_REMEDIATION_LOOPS:
        return "report"
    return "decision"


def build_graph(checkpointer: MemorySaver | None = None):
    graph = StateGraph(IncidentState)

    graph.add_node("monitor", monitor_agent)
    graph.add_node("diagnosis_agent", diagnosis_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("execution", execution_agent)
    graph.add_node("verification", verification_agent)
    graph.add_node("report", report_agent)

    graph.set_entry_point("monitor")
    graph.add_edge("monitor", "diagnosis_agent")
    graph.add_edge("diagnosis_agent", "decision")
    graph.add_edge("decision", "execution")
    graph.add_edge("execution", "verification")
    graph.add_conditional_edges(
        "verification",
        _route_after_verification,
        {"report": "report", "decision": "decision"},
    )
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
