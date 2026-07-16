"""
agents.py
---------
Each function here is a LangGraph node: it takes the current
`IncidentState`, does its work, and returns a PARTIAL dict of state
updates (LangGraph merges these using the reducers defined in state.py).

LLM backend
-----------
Nodes call `llm_client.complete(...)`, a small wrapper (see llm_client.py)
that uses a real model if `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is set
in the environment, and otherwise falls back to a deterministic
rule-based stub. This means the repository is runnable and demonstrable
with zero API keys and zero cost, while still showing the integration
point a production deployment would use. This trade-off is documented in
decision_log.md.

Every node appends an `AgentDecision` and any `ToolCall`s it makes to
state, which is what Part 1's trajectory-based evaluation pipeline
consumes.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import prompts
from llm_client import get_llm_client
from state import AgentDecision, IncidentState, IncidentStatus, ToolCall
from tools import (
    TOOLS_BY_NAME,
    check_dependency_health,
    get_metrics,
    page_oncall,
    query_logs,
    restart_service,
    rollback_deployment,
    scale_service,
    verify_slo,
)

llm = get_llm_client()


def _record_tool_call(fn, **kwargs) -> tuple[Any, ToolCall]:
    """Invoke a LangChain @tool and wrap the result/latency for trajectory logging."""
    start = time.perf_counter()
    try:
        result = fn.invoke(kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        success = not (isinstance(result, dict) and result.get("success") is False)
        return result, ToolCall(
            tool_name=fn.name,
            arguments=kwargs,
            result=result,
            success=success,
            latency_ms=latency_ms,
            error=None if success else result.get("error"),
        )
    except Exception as exc:  # tool itself raised -> hallucinated args, bad call, etc.
        latency_ms = (time.perf_counter() - start) * 1000
        return None, ToolCall(
            tool_name=fn.name,
            arguments=kwargs,
            result=None,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
        )


# ---------------------------------------------------------------------
# 1. Monitor Agent
# ---------------------------------------------------------------------

def monitor_agent(state: IncidentState) -> dict:
    alert = state["alert"]
    response = llm.complete(
        system=prompts.MONITOR_AGENT_SYSTEM_PROMPT,
        user=f"Alert: {alert}",
        task="monitor",
    )
    decision = AgentDecision(
        agent="monitor",
        reasoning=response["reasoning"],
        decision=response["decision"],
    )
    return {
        "status": IncidentStatus.MONITORING.value,
        "decisions": [decision],
        "messages": [{"role": "system", "agent": "monitor", "content": response["decision"]}],
    }


# ---------------------------------------------------------------------
# 2. Diagnosis Agent
# ---------------------------------------------------------------------

def diagnosis_agent(state: IncidentState) -> dict:
    alert = state["alert"]
    service = alert["service"]

    tool_calls: list[ToolCall] = []

    logs_result, tc1 = _record_tool_call(query_logs, service=service, minutes=15)
    tool_calls.append(tc1)

    metrics_result, tc2 = _record_tool_call(get_metrics, service=service, metric=alert["metric"])
    tool_calls.append(tc2)

    deps_result, tc3 = _record_tool_call(check_dependency_health, service=service)
    tool_calls.append(tc3)

    diagnosis = llm.diagnose(
        alert=alert,
        logs=logs_result,
        metrics=metrics_result,
        dependencies=deps_result,
    )

    decision = AgentDecision(
        agent="diagnosis",
        reasoning=diagnosis["reasoning"],
        decision=diagnosis["hypothesis"],
        confidence=diagnosis["confidence"],
    )

    memory_note = f"[{service}] hypothesis='{diagnosis['hypothesis']}' confidence={diagnosis['confidence']}"

    return {
        "status": IncidentStatus.DIAGNOSING.value,
        "diagnosis": diagnosis,
        "tool_calls": tool_calls,
        "decisions": [decision],
        "memory": [memory_note],
    }


# ---------------------------------------------------------------------
# 3. Decision Agent
# ---------------------------------------------------------------------

def decision_agent(state: IncidentState) -> dict:
    diagnosis = state["diagnosis"]
    alert = state["alert"]

    action = llm.decide_action(diagnosis=diagnosis, severity=alert["severity"], service=alert["service"])

    decision = AgentDecision(
        agent="decision",
        reasoning=action["reasoning"],
        decision=f"{action['action']}({action['arguments']})",
        confidence=action.get("confidence"),
    )

    return {
        "status": IncidentStatus.DECIDING.value,
        "chosen_action": action,
        "decisions": [decision],
    }


# ---------------------------------------------------------------------
# 4. Execution Agent (contains the failure-recovery path)
# ---------------------------------------------------------------------

_ACTION_TOOL_MAP = {
    "restart_service": restart_service,
    "scale_service": scale_service,
    "rollback_deployment": rollback_deployment,
    "page_oncall": page_oncall,
}


def execution_agent(state: IncidentState) -> dict:
    action = state["chosen_action"]
    tool_fn = _ACTION_TOOL_MAP.get(action["action"])
    tool_calls: list[ToolCall] = []
    decisions: list[AgentDecision] = []

    if tool_fn is None:
        # Hallucinated / unknown tool name from the Decision Agent.
        decisions.append(
            AgentDecision(
                agent="execution",
                reasoning=f"Decision Agent selected unknown action '{action['action']}'.",
                decision="escalate: unknown action, cannot execute",
            )
        )
        result, tc = _record_tool_call(
            page_oncall,
            service=state["alert"]["service"],
            severity="high",
            summary=f"Unrecognized automated action '{action['action']}' — needs human review.",
        )
        tool_calls.append(tc)
        return {
            "status": IncidentStatus.ESCALATED.value,
            "execution_result": result,
            "tool_calls": tool_calls,
            "decisions": decisions,
        }

    result, tc = _record_tool_call(tool_fn, **action["arguments"])
    tool_calls.append(tc)

    if not tc.success:
        decisions.append(
            AgentDecision(
                agent="execution",
                reasoning=f"First attempt at '{action['action']}' failed: {tc.error}. Retrying once per policy.",
                decision="retry",
            )
        )
        retry_result, retry_tc = _record_tool_call(tool_fn, **action["arguments"])
        tool_calls.append(retry_tc)

        if not retry_tc.success:
            decisions.append(
                AgentDecision(
                    agent="execution",
                    reasoning="Retry also failed. Escalating to human on-call per no-more-than-one-retry policy.",
                    decision="escalate",
                )
            )
            page_result, page_tc = _record_tool_call(
                page_oncall,
                service=state["alert"]["service"],
                severity=state["alert"]["severity"],
                summary=f"Automated remediation '{action['action']}' failed twice.",
            )
            tool_calls.append(page_tc)
            return {
                "status": IncidentStatus.ESCALATED.value,
                "execution_result": page_result,
                "tool_calls": tool_calls,
                "decisions": decisions,
                "retry_count": state["retry_count"] + 1,
            }

        result = retry_result
        decisions.append(
            AgentDecision(agent="execution", reasoning="Retry succeeded.", decision="proceed to verification")
        )

    return {
        "status": IncidentStatus.EXECUTING.value,
        "execution_result": result,
        "tool_calls": tool_calls,
        "decisions": decisions,
        "retry_count": state["retry_count"] + (1 if len(tool_calls) > 1 else 0),
    }


# ---------------------------------------------------------------------
# 5. Verification Agent
# ---------------------------------------------------------------------

def verification_agent(state: IncidentState) -> dict:
    if state["status"] == IncidentStatus.ESCALATED.value:
        # Nothing to verify — a human now owns the incident.
        return {"status": IncidentStatus.ESCALATED.value}

    alert = state["alert"]
    result, tc = _record_tool_call(verify_slo, service=alert["service"], metric=alert["metric"])

    resolved = bool(result and result.get("within_slo"))
    decision = AgentDecision(
        agent="verification",
        reasoning=f"Post-remediation metric value {result.get('value') if result else 'N/A'} vs SLO threshold {alert['threshold']}.",
        decision="resolved" if resolved else "not resolved — escalating",
    )

    return {
        "status": IncidentStatus.RESOLVED.value if resolved else IncidentStatus.ESCALATED.value,
        "verification_result": result,
        "tool_calls": [tc],
        "decisions": [decision],
    }


# ---------------------------------------------------------------------
# 6. Report node
# ---------------------------------------------------------------------

def report_agent(state: IncidentState) -> dict:
    trajectory_summary = {
        "alert": state["alert"],
        "diagnosis": state["diagnosis"],
        "chosen_action": state["chosen_action"],
        "execution_result": state["execution_result"],
        "verification_result": state["verification_result"],
        "decisions": [d.decision for d in state["decisions"]],
        "tool_call_count": len(state["tool_calls"]),
        "retry_count": state["retry_count"],
        "final_status": state["status"],
    }
    report = llm.synthesize_report(trajectory_summary)
    return {"final_report": report}
