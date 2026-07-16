"""
tools.py
--------
Mock "production" tools the agents call. In a real deployment these would
wrap Datadog/Prometheus, PagerDuty, Kubernetes, and a runbook store. Here
they are deterministic-ish simulations so the graph is runnable and
testable offline, with NO external network calls or API keys required.

Each tool is exposed twice:
1. As a plain Python function (used by mock/unit tests and by the
   Diagnosis/Execution agents directly).
2. As a LangChain `@tool`-decorated wrapper (used when an LLM is driving
   tool selection via function calling), so trajectory evaluation can
   inspect real `tool_call` objects in the message history.

A `FAILURE_INJECTION` map lets the POC demonstrate the "recover from tool
failure" requirement deterministically instead of relying on randomness,
which would make the demo non-reproducible.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from langchain_core.tools import tool

# Injected failures: first N calls to a tool name fail, then it succeeds.
# This makes the failure-recovery path exercised on every run without
# being flaky.
FAILURE_INJECTION = {
    "restart_service": 1,  # first call fails, retry succeeds
}
_call_counts: dict[str, int] = {}


def _simulate_latency(min_ms: int = 40, max_ms: int = 220) -> float:
    delay = random.uniform(min_ms, max_ms) / 1000
    time.sleep(delay)
    return delay * 1000


def _should_fail(tool_name: str) -> bool:
    _call_counts[tool_name] = _call_counts.get(tool_name, 0) + 1
    return _call_counts[tool_name] <= FAILURE_INJECTION.get(tool_name, 0)


# ---------------------------------------------------------------------
# Diagnosis-time tools
# ---------------------------------------------------------------------

@tool
def query_logs(service: str, minutes: int = 15) -> dict:
    """Fetch recent error logs for a service from the log aggregator."""
    _simulate_latency()
    canned_errors = {
        "checkout-api": [
            "ConnectionPoolTimeoutError: pool exhausted after 30s",
            "psycopg2.OperationalError: too many connections",
        ],
        "payments-worker": [
            "OOMKilled: container exceeded 512Mi memory limit",
        ],
        "notification-service": [
            "NullPointerException in notification dispatch handler",
        ],
    }
    return {
        "service": service,
        "window_minutes": minutes,
        "error_lines": canned_errors.get(service, []),
    }


@tool
def get_metrics(service: str, metric: str) -> dict:
    """Fetch the current value and 1-hour trend for a service metric."""
    _simulate_latency()
    return {
        "service": service,
        "metric": metric,
        "current_value": round(random.uniform(70, 99), 1),
        "trend": "increasing",
    }


@tool
def check_dependency_health(service: str) -> dict:
    """Check health of a service's declared upstream dependencies."""
    _simulate_latency()
    return {
        "service": service,
        "dependencies": {
            "database": "degraded" if service == "checkout-api" else "healthy",
            "cache": "healthy",
            "queue": "healthy",
        },
    }


# ---------------------------------------------------------------------
# Execution-time tools
# ---------------------------------------------------------------------

@tool
def restart_service(service: str) -> dict:
    """Restart a service's pods/instances via the orchestrator."""
    latency = _simulate_latency(100, 300)
    if _should_fail("restart_service"):
        return {
            "service": service,
            "success": False,
            "error": "OrchestratorTimeoutError: rollout status check timed out",
            "latency_ms": latency,
        }
    return {"service": service, "success": True, "latency_ms": latency}


@tool
def scale_service(service: str, replicas: int) -> dict:
    """Scale a service to a target replica count."""
    latency = _simulate_latency()
    return {"service": service, "replicas": replicas, "success": True, "latency_ms": latency}


@tool
def rollback_deployment(service: str, to_version: Optional[str] = None) -> dict:
    """Roll a service back to its last known-good deployment."""
    latency = _simulate_latency(150, 350)
    return {
        "service": service,
        "rolled_back_to": to_version or "previous-stable",
        "success": True,
        "latency_ms": latency,
    }


@tool
def page_oncall(service: str, severity: str, summary: str) -> dict:
    """Escalate to a human on-call engineer when automated remediation is unsafe."""
    _simulate_latency(20, 80)
    return {"service": service, "paged": True, "severity": severity, "summary": summary}


# ---------------------------------------------------------------------
# Verification-time tools
# ---------------------------------------------------------------------

@tool
def verify_slo(service: str, metric: str) -> dict:
    """Re-check the triggering metric against its SLO threshold post-remediation."""
    _simulate_latency()
    recovered_value = round(random.uniform(20, 55), 1)
    return {"service": service, "metric": metric, "value": recovered_value, "within_slo": True}


ALL_TOOLS = [
    query_logs,
    get_metrics,
    check_dependency_health,
    restart_service,
    scale_service,
    rollback_deployment,
    page_oncall,
    verify_slo,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
