"""
evaluation/trajectory_eval.py
------------------------------
A runnable implementation of the trajectory-based evaluation methodology
described in report/research_report.md, applied to THIS repo's own
incident-response graph (not to GPT-4.1/Qwen3, since we don't have API
access to both in this environment — see the report for why the
methodology is designed to be model-agnostic and pluggable).

This scores a single run's trajectory (the full IncidentState produced
by app.py) against the metric families the research report defines:

  - planning_quality       : did diagnosis precede action, in order?
  - tool_selection_accuracy: were the tools called appropriate to the
                              diagnosed root cause (rule-based check
                              against an "expected tool" policy table)?
  - recovery_from_failure  : if a tool call failed, was a retry or
                              escalation attempted (not silently dropped)?
  - hallucinated_tool_rate : fraction of tool calls referencing a tool
                              name not in the registered tool set
  - state_consistency      : did state fields get set in the expected
                              monotonic order (no skipped stages)?
  - task_success            : did the run end RESOLVED?
  - latency_ms              : total wall-clock time across tool calls
  - step_efficiency         : tool_calls / minimum_plausible_tool_calls

This is intentionally a RUBRIC + rule-based scorer, not an "LLM-judges-
itself" black box — every score is traceable to a concrete field in the
trajectory, which is what makes trajectory evaluation auditable (see
report section 2.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_TOOL_NAMES = {
    "query_logs",
    "get_metrics",
    "check_dependency_health",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "page_oncall",
    "verify_slo",
}

# Minimum tool calls a competent agent needs: 3 diagnosis tools + >=1
# execution tool + 1 verification tool (page_oncall paths skip verify_slo).
MIN_PLAUSIBLE_TOOL_CALLS = 4

EXPECTED_STAGE_ORDER = [
    "received",
    "monitoring",
    "diagnosing",
    "deciding",
    "executing",
    "verifying",  # not a literal status value in this POC but conceptually here
]


@dataclass
class TrajectoryScore:
    task_success: bool
    planning_quality: float
    tool_selection_accuracy: float
    recovery_from_failure: float
    hallucinated_tool_rate: float
    state_consistency: float
    total_latency_ms: float
    tool_call_count: int
    step_efficiency: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_success": self.task_success,
            "planning_quality": round(self.planning_quality, 3),
            "tool_selection_accuracy": round(self.tool_selection_accuracy, 3),
            "recovery_from_failure": round(self.recovery_from_failure, 3),
            "hallucinated_tool_rate": round(self.hallucinated_tool_rate, 3),
            "state_consistency": round(self.state_consistency, 3),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "tool_call_count": self.tool_call_count,
            "step_efficiency": round(self.step_efficiency, 3),
            "notes": self.notes,
        }


def score_trajectory(trajectory: dict) -> TrajectoryScore:
    tool_calls = trajectory["tool_calls"]
    decisions = trajectory["decisions"]
    notes: list[str] = []

    # --- task_success ---
    task_success = trajectory["status"] in ("resolved",)

    # --- hallucinated_tool_rate ---
    hallucinated = [tc for tc in tool_calls if tc["tool_name"] not in VALID_TOOL_NAMES]
    hallucinated_tool_rate = len(hallucinated) / len(tool_calls) if tool_calls else 0.0
    if hallucinated:
        notes.append(f"{len(hallucinated)} call(s) referenced unregistered tools: {[h['tool_name'] for h in hallucinated]}")

    # --- planning_quality: diagnosis tools must occur before any execution tool ---
    diagnosis_tools = {"query_logs", "get_metrics", "check_dependency_health"}
    execution_tools = {"restart_service", "scale_service", "rollback_deployment", "page_oncall"}
    first_exec_idx = next((i for i, tc in enumerate(tool_calls) if tc["tool_name"] in execution_tools), None)
    if first_exec_idx is None:
        planning_quality = 0.5  # no execution attempted at all — can't assess ordering
        notes.append("No execution tool calls found; planning ordering unscored.")
    else:
        diag_before = [tc for tc in tool_calls[:first_exec_idx] if tc["tool_name"] in diagnosis_tools]
        planning_quality = min(1.0, len(diag_before) / 2)  # expect >=2 diagnosis calls before acting
        if not diag_before:
            notes.append("Execution attempted with zero prior diagnosis tool calls.")

    # --- tool_selection_accuracy: rule-based policy check ---
    # Policy (from prompts.py DECISION_AGENT_SYSTEM_PROMPT): rollback only if
    # diagnosis implicates a deployment; page_oncall only for critical/low-confidence.
    diagnosis = trajectory.get("diagnosis") or {}
    hypothesis = (diagnosis.get("hypothesis") or "").lower()
    exec_calls = [tc for tc in tool_calls if tc["tool_name"] in execution_tools]
    violations = 0
    for tc in exec_calls:
        if tc["tool_name"] == "rollback_deployment" and "deploy" not in hypothesis:
            violations += 1
            notes.append("rollback_deployment called without deployment-related evidence in diagnosis.")
    tool_selection_accuracy = 1.0 if not exec_calls else max(0.0, 1 - violations / len(exec_calls))

    # --- recovery_from_failure ---
    failed_calls = [tc for tc in tool_calls if not tc["success"]]
    if not failed_calls:
        recovery_from_failure = 1.0  # nothing to recover from -> vacuously fine, flagged in notes
        notes.append("No tool failures occurred in this run; recovery path not exercised.")
    else:
        # A failure counts as "recovered" if a later call either retries the same
        # tool successfully or escalates via page_oncall.
        recovered = 0
        for i, tc in enumerate(tool_calls):
            if not tc["success"]:
                later = tool_calls[i + 1 :]
                if any(t["tool_name"] == tc["tool_name"] and t["success"] for t in later) or any(
                    t["tool_name"] == "page_oncall" for t in later
                ):
                    recovered += 1
        recovery_from_failure = recovered / len(failed_calls)

    # --- state_consistency ---
    # Rough proxy: decisions list should reference each agent role in order.
    agent_order = [d["agent"] for d in decisions]
    expected_roles = ["monitor", "diagnosis", "decision", "execution"]
    seen_order = [a for a in agent_order if a in expected_roles]
    state_consistency = 1.0 if seen_order == sorted(seen_order, key=lambda a: expected_roles.index(a)) else 0.5

    total_latency_ms = sum(tc["latency_ms"] for tc in tool_calls)
    step_efficiency = MIN_PLAUSIBLE_TOOL_CALLS / len(tool_calls) if tool_calls else 0.0

    return TrajectoryScore(
        task_success=task_success,
        planning_quality=planning_quality,
        tool_selection_accuracy=tool_selection_accuracy,
        recovery_from_failure=recovery_from_failure,
        hallucinated_tool_rate=hallucinated_tool_rate,
        state_consistency=state_consistency,
        total_latency_ms=total_latency_ms,
        tool_call_count=len(tool_calls),
        step_efficiency=step_efficiency,
        notes=notes,
    )


def score_run_file(path: str) -> dict[str, dict]:
    """Score every trajectory in a JSON file produced by `python app.py --json out.json`."""
    import json

    with open(path) as f:
        runs = json.load(f)
    return {name: score_trajectory(traj).as_dict() for name, traj in runs.items()}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluation/trajectory_eval.py <trajectories.json>")
        raise SystemExit(1)
    scores = score_run_file(sys.argv[1])
    import json as _json

    print(_json.dumps(scores, indent=2))
