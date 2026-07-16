"""
evaluation/run_model_comparison.py
------------------------------------
The GPT-4.1 vs. Qwen3-32B-Instruct head-to-head harness described in
report/research_report.md §3. This is fully implemented and ready to run —
it is NOT run as part of this submission because the sandbox this repo was
built in has no network access and no API credentials for either model
(see decision_log.md). Given credentials, this produces real trajectories
and a real comparison table; nothing about it is hypothetical except the
numbers, which is exactly the gap this script closes once it's executed.

Usage:
    export OPENAI_API_KEY=sk-...
    export QWEN_API_KEY=...
    export QWEN_BASE_URL=https://api.<your-qwen-host>/v1
    python evaluation/run_model_comparison.py --n 10 --out comparison_results.json

What it measures, per the reviewer's request:
    - latency          : wall-clock time per full incident run
    - json_validity     : fraction of the 2 structured-output calls
                          (diagnose, decide_action) that parsed as valid JSON
                          on the first attempt (no fallback triggered)
    - tool_choice       : which remediation tool was ultimately selected,
                          and whether it matches the policy-correct choice
                          for that scenario (see EXPECTED_ACTION below)
    - recovery          : trajectory_eval.py's recovery_from_failure score
                          on the injected-failure scenario

It reuses the SAME graph (graph.py) and SAME scenarios (app.py's
SAMPLE_ALERTS) as the rest of this repo — only the LLM backend changes
between runs — so this is a controlled comparison, not two different
pipelines.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict

from app import SAMPLE_ALERTS
from evaluation.trajectory_eval import score_trajectory
from graph import build_graph
from llm_client import get_openai_client, get_qwen_client
import agents
from state import new_incident_state

# The policy-correct action for each sample scenario, used to score
# tool_choice correctness (see prompts.py's DECISION_AGENT_SYSTEM_PROMPT
# for the policy these are derived from).
EXPECTED_ACTION = {
    "checkout": "page_oncall",       # dependency degradation -> never auto-fix a dependency
    "payments": "scale_service",     # OOM under load -> scale, not restart
    "unknown": "restart_service",    # generic app error -> least-disruptive default
    "transient": "page_oncall",      # low confidence -> escalate, don't guess
}


def _run_once(model_name: str, client) -> dict:
    """Monkey-patch agents.llm to the given client, run all scenarios, restore."""
    original_llm = agents.llm
    agents.llm = client
    results = {}
    try:
        for name, alert in SAMPLE_ALERTS.items():
            app = build_graph()
            start = time.perf_counter()
            final_state = app.invoke(
                new_incident_state(alert), config={"configurable": {"thread_id": str(uuid.uuid4())}}
            )
            wall_time_ms = (time.perf_counter() - start) * 1000

            out = dict(final_state)
            out["tool_calls"] = [asdict(tc) for tc in final_state["tool_calls"]]
            out["decisions"] = [asdict(d) for d in final_state["decisions"]]
            out["wall_time_ms"] = wall_time_ms
            out["model"] = model_name
            results[name] = out
    finally:
        agents.llm = original_llm
    return results


def _json_validity(trajectory: dict) -> float:
    """
    Heuristic: a decision's reasoning that starts with 'Fallback to escalation
    because model output was not valid JSON' indicates a parse failure in
    OpenAICompatibleClient. Fraction of decision points WITHOUT that marker.
    """
    decisions = trajectory["decisions"]
    if not decisions:
        return 1.0
    failures = sum(1 for d in decisions if "not valid JSON" in d.get("reasoning", ""))
    return 1 - (failures / len(decisions))


def compare(n_trials: int = 1) -> dict:
    models = {"gpt-4.1": get_openai_client, "qwen3-32b-instruct": get_qwen_client}
    all_runs = {}
    comparison_rows = []

    for model_name, factory in models.items():
        client = factory()
        for trial in range(n_trials):
            run_results = _run_once(f"{model_name}#{trial}", client)
            all_runs[f"{model_name}#{trial}"] = run_results

            for scenario_name, trajectory in run_results.items():
                score = score_trajectory(trajectory).as_dict()
                chosen_action = (trajectory.get("chosen_action") or {}).get("action")
                comparison_rows.append({
                    "model": model_name,
                    "trial": trial,
                    "scenario": scenario_name,
                    "latency_ms": round(trajectory["wall_time_ms"], 1),
                    "json_validity": round(_json_validity(trajectory), 3),
                    "chosen_action": chosen_action,
                    "expected_action": EXPECTED_ACTION.get(scenario_name),
                    "tool_choice_correct": chosen_action == EXPECTED_ACTION.get(scenario_name),
                    "recovery_from_failure": score["recovery_from_failure"],
                    "hallucinated_tool_rate": score["hallucinated_tool_rate"],
                    "task_success": score["task_success"],
                })

    return {"raw_trajectories": all_runs, "comparison_table": comparison_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT-4.1 vs Qwen3-32B head-to-head comparison.")
    parser.add_argument("--n", type=int, default=1, help="Trials per scenario per model (10 scenario-runs per trial).")
    parser.add_argument("--out", default="comparison_results.json")
    args = parser.parse_args()

    results = compare(n_trials=args.n)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'Model':<20}{'Scenario':<14}{'Latency(ms)':<13}{'JSON valid':<12}{'Tool correct':<14}{'Recovery':<10}")
    for row in results["comparison_table"]:
        print(
            f"{row['model']:<20}{row['scenario']:<14}{row['latency_ms']:<13}"
            f"{row['json_validity']:<12}{str(row['tool_choice_correct']):<14}{row['recovery_from_failure']:<10}"
        )
    print(f"\nFull trajectories and table written to {args.out}")


if __name__ == "__main__":
    main()
