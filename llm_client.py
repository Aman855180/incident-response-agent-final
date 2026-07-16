"""
llm_client.py
-------------
A thin abstraction so agents.py never talks to a specific SDK directly.

`get_llm_client()` returns:
- `AnthropicLLMClient` if `ANTHROPIC_API_KEY` is set, or
- `MockLLMClient` otherwise (default — no key required to run the demo).

The mock is intentionally RULE-BASED, not random-fabrication: it inspects
the actual tool outputs passed to it and produces a deterministic,
inspectable hypothesis/action. This lets the graph, the failure-recovery
path, and the evaluation pipeline all be demonstrated and unit-tested
without an API key or network access, which matters for a take-home
reviewer running this locally. See decision_log.md for why this design
was chosen over hard-requiring a live model.

Swapping in a real model only requires implementing the same four
methods (`complete`, `diagnose`, `decide_action`, `synthesize_report`)
against whichever SDK you prefer — the graph and agents are unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Optional


class BaseLLMClient:
    def complete(self, system: str, user: str, task: str) -> dict:
        raise NotImplementedError

    def diagnose(self, alert: dict, logs: dict, metrics: dict, dependencies: dict) -> dict:
        raise NotImplementedError

    def decide_action(self, diagnosis: dict, severity: str, service: str = "unknown") -> dict:
        raise NotImplementedError

    def synthesize_report(self, trajectory: dict) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Deterministic, rule-based stand-in for an LLM. No network calls."""

    def complete(self, system: str, user: str, task: str) -> dict:
        return {
            "reasoning": "Alert crosses configured threshold and matches a known-service pattern; treating as genuine.",
            "decision": "Genuine incident. Proceeding to diagnosis.",
        }

    def diagnose(self, alert: dict, logs: dict, metrics: dict, dependencies: dict) -> dict:
        error_lines = logs.get("error_lines", []) if logs else []
        deps = dependencies.get("dependencies", {}) if dependencies else {}
        degraded_deps = [k for k, v in deps.items() if v != "healthy"]

        if degraded_deps:
            hypothesis = f"Upstream dependency degradation ({', '.join(degraded_deps)}) is the likely root cause."
            confidence = 0.75
            reasoning = (
                f"Logs show {len(error_lines)} matching error signature(s); "
                f"dependency check flags {degraded_deps} as degraded, which correlates "
                f"with the elevated '{alert['metric']}' metric."
            )
        elif any("OOM" in line for line in error_lines):
            hypothesis = "Resource exhaustion (OOMKilled) — container memory limit too low for current load."
            confidence = 0.8
            reasoning = "Log line explicitly reports OOMKilled; no dependency degradation observed."
        elif error_lines:
            hypothesis = "Application-level error surfaced in logs; restarting is the standard first remediation."
            confidence = 0.6
            reasoning = f"Errors present ({error_lines[0]}) matching a known transient-fault pattern; no dependency degradation observed."
        else:
            hypothesis = "No strong evidence found; likely a transient spike or false-positive alert."
            confidence = 0.3
            reasoning = "No matching error signatures and all dependencies report healthy."

        return {"hypothesis": hypothesis, "confidence": confidence, "reasoning": reasoning}

    def decide_action(self, diagnosis: dict, severity: str, service: str = "unknown") -> dict:
        confidence = diagnosis.get("confidence", 0)
        hypothesis = diagnosis.get("hypothesis", "")

        if severity == "critical" or confidence < 0.5:
            return {
                "action": "page_oncall",
                "arguments": {
                    "service": service,
                    "severity": severity,
                    "summary": f"Low-confidence or critical incident: {hypothesis}",
                },
                "reasoning": "Policy requires human escalation for critical severity or low-confidence diagnosis.",
                "confidence": confidence,
            }
        if "dependency" in hypothesis.lower():
            return {
                "action": "page_oncall",
                "arguments": {
                    "service": service,
                    "severity": severity,
                    "summary": f"Upstream dependency issue, outside blast radius of automated remediation: {hypothesis}",
                },
                "reasoning": "Dependency-caused incidents are not safely fixable by restarting/scaling the dependent service.",
                "confidence": confidence,
            }
        if "OOM" in hypothesis or "resource exhaustion" in hypothesis.lower():
            return {
                "action": "scale_service",
                "arguments": {"service": service, "replicas": 4},
                "reasoning": "OOM under load is best addressed by scaling out/up rather than a bare restart.",
                "confidence": confidence,
            }
        return {
            "action": "restart_service",
            "arguments": {"service": service},
            "reasoning": "Default least-disruptive remediation when no specific pattern is matched.",
            "confidence": confidence,
        }

    def synthesize_report(self, trajectory: dict) -> str:
        alert = trajectory["alert"]
        lines = [
            f"# Incident Report — {alert['alert_id']}",
            "",
            f"**Service:** {alert['service']}  ",
            f"**Triggering metric:** {alert['metric']} = {alert['value']} (threshold {alert['threshold']})  ",
            f"**Severity:** {alert['severity']}  ",
            f"**Final status:** {trajectory['final_status']}",
            "",
            "## Root Cause",
            trajectory["diagnosis"]["hypothesis"] if trajectory["diagnosis"] else "Not determined.",
            "",
            "## Actions Taken",
        ]
        for d in trajectory["decisions"]:
            lines.append(f"- {d}")
        lines += [
            "",
            f"**Tool calls made:** {trajectory['tool_call_count']}  ",
            f"**Retries:** {trajectory['retry_count']}",
            "",
            "## Verification",
            str(trajectory["verification_result"]),
        ]
        return "\n".join(lines)


class AnthropicLLMClient(BaseLLMClient):
    """
    Real-model backend. Only imports the Anthropic SDK lazily so the
    package is not a hard dependency for running the mock-mode demo.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # noqa: F401  (imported here, not at module load)

        self._client = anthropic.Anthropic()
        self._model = model

    def _call(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def complete(self, system: str, user: str, task: str) -> dict:
        text = self._call(system, user)
        return {"reasoning": text, "decision": text}

    def diagnose(self, alert: dict, logs: dict, metrics: dict, dependencies: dict) -> dict:
        import json as _json

        prompt = (
            f"Alert: {alert}\nLogs: {logs}\nMetrics: {metrics}\nDependencies: {dependencies}\n\n"
            "Return a JSON object with keys hypothesis, confidence (0-1), reasoning."
        )
        text = self._call("You are an SRE diagnosis assistant. Respond ONLY with valid JSON.", prompt)
        try:
            return _json.loads(text)
        except Exception:
            return {"hypothesis": text, "confidence": 0.5, "reasoning": "Unstructured model output; see hypothesis."}

    def decide_action(self, diagnosis: dict, severity: str, service: str = "unknown") -> dict:
        import json as _json

        prompt = (
            f"Diagnosis: {diagnosis}\nSeverity: {severity}\nService: {service}\n\n"
            "Return ONLY valid JSON: {action, arguments, reasoning, confidence}."
        )
        text = self._call("You are an SRE decision assistant. Respond ONLY with valid JSON.", prompt)
        try:
            return _json.loads(text)
        except Exception:
            return {
                "action": "page_oncall",
                "arguments": {"service": service, "severity": severity, "summary": "Unparseable model output"},
                "reasoning": "Fallback to escalation because model output was not valid JSON.",
                "confidence": 0.0,
            }

    def synthesize_report(self, trajectory: dict) -> str:
        return self._call(
            "You write concise SRE incident reports for a postmortem audience.",
            f"Trajectory: {trajectory}",
        )


class OpenAICompatibleClient(BaseLLMClient):
    """
    Shared implementation for any OpenAI-Chat-Completions-compatible backend —
    used for both GPT-4.1 (api.openai.com) and Qwen3-32B-Instruct (served via
    an OpenAI-compatible endpoint, e.g. Together AI / Fireworks / DeepInfra /
    a self-hosted vLLM server). This is the harness referenced in
    report/research_report.md §3 — implementing it doesn't require credentials,
    only running it does. Point `base_url` at whichever Qwen3 host you have
    access to; the same class serves both sides of the comparison so latency
    and behavior are measured with an identical code path.
    """

    def __init__(self, model: str, api_key_env: str, base_url: Optional[str] = None):
        import openai  # noqa: F401  (imported here, not at module load)

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set")
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _call(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str, task: str) -> dict:
        text = self._call(system, user)
        return {"reasoning": text, "decision": text}

    def diagnose(self, alert: dict, logs: dict, metrics: dict, dependencies: dict) -> dict:
        import json as _json

        prompt = (
            f"Alert: {alert}\nLogs: {logs}\nMetrics: {metrics}\nDependencies: {dependencies}\n\n"
            "Return a JSON object with keys hypothesis, confidence (0-1), reasoning."
        )
        text = self._call(
            "You are an SRE diagnosis assistant. Respond ONLY with valid JSON.", prompt, json_mode=True
        )
        try:
            return _json.loads(text)
        except Exception:
            return {"hypothesis": text, "confidence": 0.5, "reasoning": "Unstructured model output; see hypothesis."}

    def decide_action(self, diagnosis: dict, severity: str, service: str = "unknown") -> dict:
        import json as _json

        prompt = (
            f"Diagnosis: {diagnosis}\nSeverity: {severity}\nService: {service}\n\n"
            "Return ONLY valid JSON: {action, arguments, reasoning, confidence}."
        )
        text = self._call(
            "You are an SRE decision assistant. Respond ONLY with valid JSON.", prompt, json_mode=True
        )
        try:
            return _json.loads(text)
        except Exception:
            return {
                "action": "page_oncall",
                "arguments": {"service": service, "severity": severity, "summary": "Unparseable model output"},
                "reasoning": "Fallback to escalation because model output was not valid JSON.",
                "confidence": 0.0,
            }

    def synthesize_report(self, trajectory: dict) -> str:
        return self._call(
            "You write concise SRE incident reports for a postmortem audience.",
            f"Trajectory: {trajectory}",
        )


def get_openai_client() -> OpenAICompatibleClient:
    """GPT-4.1 via the standard OpenAI API. Requires OPENAI_API_KEY."""
    return OpenAICompatibleClient(model="gpt-4.1", api_key_env="OPENAI_API_KEY")


def get_qwen_client(base_url: Optional[str] = None) -> OpenAICompatibleClient:
    """
    Qwen3-32B-Instruct via any OpenAI-compatible host. Requires QWEN_API_KEY
    and either QWEN_BASE_URL in the environment or an explicit base_url
    argument (e.g. a Together AI / Fireworks / DeepInfra / self-hosted vLLM
    endpoint serving 'Qwen/Qwen3-32B-Instruct').
    """
    resolved_base_url = base_url or os.environ.get("QWEN_BASE_URL")
    if not resolved_base_url:
        raise RuntimeError("QWEN_BASE_URL not set and no base_url provided")
    return OpenAICompatibleClient(model="Qwen/Qwen3-32B-Instruct", api_key_env="QWEN_API_KEY", base_url=resolved_base_url)


def get_llm_client() -> BaseLLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMClient()
        except Exception:
            # SDK not installed or client failed to init — fall back to mock
            # so the graph remains runnable.
            return MockLLMClient()
    return MockLLMClient()
