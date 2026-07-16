"""
app.py
------
CLI entrypoint. Runs one or more mock telemetry alerts through the
LangGraph workflow and prints the final incident report and a summary
of the trajectory (useful for eyeballing before running the full
evaluation pipeline in evaluation/).

Usage:
    python app.py                     # run all sample alerts
    python app.py --alert checkout    # run a single named sample alert
    python app.py --json out.json     # also dump full trajectories to JSON
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict

from graph import build_graph
from state import IncidentState, TelemetryAlert, new_incident_state

SAMPLE_ALERTS: dict[str, TelemetryAlert] = {
    "checkout": {
        "alert_id": "ALT-1001",
        "service": "checkout-api",
        "metric": "p99_latency_ms",
        "value": 4200.0,
        "threshold": 800.0,
        "severity": "high",
        "raw_payload": {"source": "prometheus", "rule": "checkout-latency-p99"},
    },
    "payments": {
        "alert_id": "ALT-1002",
        "service": "payments-worker",
        "metric": "memory_usage_pct",
        "value": 98.5,
        "threshold": 90.0,
        "severity": "high",
        "raw_payload": {"source": "prometheus", "rule": "payments-oom-risk"},
    },
    "unknown": {
        "alert_id": "ALT-1003",
        "service": "notification-service",
        "metric": "error_rate_pct",
        "value": 12.0,
        "threshold": 5.0,
        "severity": "medium",
        "raw_payload": {"source": "prometheus", "rule": "generic-error-rate"},
    },
    "transient": {
        "alert_id": "ALT-1004",
        "service": "reporting-service",
        "metric": "queue_depth",
        "value": 150.0,
        "threshold": 100.0,
        "severity": "low",
        "raw_payload": {"source": "prometheus", "rule": "reporting-queue-depth"},
    },
}


def _serialize(state: IncidentState) -> dict:
    out = dict(state)
    out["tool_calls"] = [asdict(tc) for tc in state["tool_calls"]]
    out["decisions"] = [asdict(d) for d in state["decisions"]]
    return out


def run_alert(name: str, alert: TelemetryAlert) -> dict:
    app = build_graph()
    initial_state = new_incident_state(alert)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final_state = app.invoke(initial_state, config=config)

    print(f"\n{'=' * 70}\nAlert: {name} ({alert['alert_id']})\n{'=' * 70}")
    print(final_state["final_report"])
    print(
        f"\n[trajectory] {len(final_state['tool_calls'])} tool calls, "
        f"{len(final_state['decisions'])} decisions, "
        f"{final_state['retry_count']} retries, "
        f"final status={final_state['status']}"
    )
    return _serialize(final_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous incident response demo.")
    parser.add_argument("--alert", choices=list(SAMPLE_ALERTS.keys()), help="Run a single sample alert.")
    parser.add_argument("--json", help="Path to write full trajectories as JSON.")
    args = parser.parse_args()

    alerts = {args.alert: SAMPLE_ALERTS[args.alert]} if args.alert else SAMPLE_ALERTS
    results = {name: run_alert(name, alert) for name, alert in alerts.items()}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote full trajectories to {args.json}")


if __name__ == "__main__":
    main()
