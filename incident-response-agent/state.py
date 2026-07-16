"""
state.py
--------
Typed state definitions shared across the LangGraph workflow.

Design notes (see decision_log.md for the full rationale):
- We use a single `TypedDict` as the graph state because LangGraph's
  `StateGraph` merges partial dict updates from each node automatically,
  which is the framework's native mental model. A dataclass would require
  a custom reducer for every field; TypedDict + `Annotated` reducers is
  the idiomatic LangGraph pattern as of the 0.2.x API.
- Every field that should ACCUMULATE across nodes (logs, tool calls,
  memory) uses `operator.add` as its reducer so nodes can return only the
  new items instead of the full accumulated list.
- Every field that should be OVERWRITTEN (current status, diagnosis)
  has no reducer, so the latest node's value wins.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    RECEIVED = "received"
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    DECIDING = "deciding"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class ToolCall:
    """A single tool invocation, kept for trajectory-based evaluation."""

    tool_name: str
    arguments: dict
    result: Any
    success: bool
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None


@dataclass
class AgentDecision:
    """A single reasoning/decision checkpoint, kept for trajectory eval."""

    agent: str
    reasoning: str
    decision: str
    confidence: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TelemetryAlert(TypedDict):
    alert_id: str
    service: str
    metric: str
    value: float
    threshold: float
    severity: str
    raw_payload: dict


class IncidentState(TypedDict):
    """
    The full graph state. Every agent node reads a subset of this and
    returns a partial dict; LangGraph merges it using the reducers below.
    """

    # --- Immutable input ---
    alert: TelemetryAlert

    # --- Mutable "latest value wins" fields (no reducer -> overwrite) ---
    status: str  # IncidentStatus value
    diagnosis: Optional[dict]  # structured root-cause hypothesis
    chosen_action: Optional[dict]  # the action Decision Agent selected
    execution_result: Optional[dict]
    verification_result: Optional[dict]
    final_report: Optional[str]
    retry_count: int

    # --- Accumulating fields (Annotated with operator.add -> append) ---
    tool_calls: Annotated[list[ToolCall], operator.add]
    decisions: Annotated[list[AgentDecision], operator.add]
    messages: Annotated[list[dict], operator.add]  # LLM chat trajectory
    memory: Annotated[list[str], operator.add]  # cross-run episodic notes


def new_incident_state(alert: TelemetryAlert) -> IncidentState:
    """Factory for a fresh state given an incoming alert."""
    return IncidentState(
        alert=alert,
        status=IncidentStatus.RECEIVED.value,
        diagnosis=None,
        chosen_action=None,
        execution_result=None,
        verification_result=None,
        final_report=None,
        retry_count=0,
        tool_calls=[],
        decisions=[],
        messages=[],
        memory=[],
    )
