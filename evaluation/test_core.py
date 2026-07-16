"""
evaluation/test_core.py
------------------------
Unit tests for the framework-independent logic: tools.py's mock APIs,
the failure-injection mechanism, and trajectory_eval.py's scoring rules.

These deliberately do NOT import langgraph/langchain so they can run in
any environment, including one without those packages installed, to
verify business logic in isolation from orchestration.

Run with: pytest evaluation/test_core.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.trajectory_eval import score_trajectory


def _base_trajectory(**overrides) -> dict:
    base = {
        "status": "resolved",
        "diagnosis": {"hypothesis": "Resource exhaustion (OOMKilled)", "confidence": 0.8},
        "tool_calls": [
            {"tool_name": "query_logs", "success": True, "latency_ms": 100},
            {"tool_name": "get_metrics", "success": True, "latency_ms": 90},
            {"tool_name": "check_dependency_health", "success": True, "latency_ms": 80},
            {"tool_name": "scale_service", "success": True, "latency_ms": 150},
            {"tool_name": "verify_slo", "success": True, "latency_ms": 60},
        ],
        "decisions": [
            {"agent": "monitor", "decision": "genuine"},
            {"agent": "diagnosis", "decision": "OOM"},
            {"agent": "decision", "decision": "scale_service"},
            {"agent": "execution", "decision": "proceed"},
        ],
    }
    base.update(overrides)
    return base


def test_successful_run_scores_well():
    score = score_trajectory(_base_trajectory()).as_dict()
    assert score["task_success"] is True
    assert score["planning_quality"] == 1.0
    assert score["hallucinated_tool_rate"] == 0.0


def test_hallucinated_tool_is_detected():
    traj = _base_trajectory()
    traj["tool_calls"].append({"tool_name": "delete_production_database", "success": True, "latency_ms": 10})
    score = score_trajectory(traj).as_dict()
    assert score["hallucinated_tool_rate"] > 0
    assert any("unregistered" in n for n in score["notes"])


def test_planning_penalizes_acting_before_diagnosis():
    traj = _base_trajectory(
        tool_calls=[
            {"tool_name": "scale_service", "success": True, "latency_ms": 150},
            {"tool_name": "query_logs", "success": True, "latency_ms": 100},
        ]
    )
    score = score_trajectory(traj).as_dict()
    assert score["planning_quality"] == 0.0


def test_recovery_scores_when_retry_follows_failure():
    traj = _base_trajectory(
        tool_calls=[
            {"tool_name": "restart_service", "success": False, "latency_ms": 200},
            {"tool_name": "restart_service", "success": True, "latency_ms": 180},
        ]
    )
    score = score_trajectory(traj).as_dict()
    assert score["recovery_from_failure"] == 1.0


def test_recovery_scores_zero_when_failure_is_dropped():
    traj = _base_trajectory(
        tool_calls=[
            {"tool_name": "restart_service", "success": False, "latency_ms": 200},
        ]
    )
    score = score_trajectory(traj).as_dict()
    assert score["recovery_from_failure"] == 0.0


def test_rollback_without_deploy_evidence_is_a_policy_violation():
    traj = _base_trajectory(
        diagnosis={"hypothesis": "Resource exhaustion (OOMKilled)", "confidence": 0.8},
        tool_calls=[
            {"tool_name": "query_logs", "success": True, "latency_ms": 100},
            {"tool_name": "get_metrics", "success": True, "latency_ms": 90},
            {"tool_name": "rollback_deployment", "success": True, "latency_ms": 150},
        ],
    )
    score = score_trajectory(traj).as_dict()
    assert score["tool_selection_accuracy"] < 1.0


if __name__ == "__main__":
    import subprocess

    subprocess.run(["python3", "-m", "pytest", __file__, "-v"])
