#!/usr/bin/env python3
"""LLM-backed triage agent for CoreSmith headless escalation packets."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM
from orchestrator.utils import atomic_write, parse_llm_json


SYSTEM = """You are the CoreSmith headless triage/debug agent.

You are not the generator and you are not the human. Your job is to read the
escalation packet and relevant on-disk OTEL/log/design artifacts, then choose
one explicit next action from the packet's allowed_actions.

Rules:
- Do not modify files.
- Do not auto-accept or rubber-stamp retries.
- Read the referenced logs/artifacts first, including OTEL/pipeline events,
  step logs, RTL, testbench, VCD/WaveKit audit, ERS, and uarch specs when
  present.
- Classify the root cause and cite concrete evidence.
- If the evidence is inconclusive or the KPI/ERS requirement cannot be
  verified, choose an action that escalates/stops rather than guessing.
- Return only a JSON object.
"""


def _load_packet(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        packet = json.load(fh)
    if not isinstance(packet, dict):
        raise ValueError(f"Escalation packet is not a JSON object: {path}")
    return packet


def _fallback_decision(packet: dict, reason: str) -> dict:
    allowed = packet.get("allowed_actions") or []
    action = "abort" if "abort" in allowed else allowed[0] if allowed else "abort"
    return {
        "action": action,
        "rationale": reason,
        "evidence": [],
        "confidence": 0.0,
        "triage_agent_error": True,
    }


def _is_transient_llm_failure(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "usage limit",
            "rate limit",
            "try again",
            "temporarily unavailable",
            "overloaded",
            "timeout",
        )
    )


async def triage(path: Path, dry_run: bool = False) -> dict:
    packet = _load_packet(path)
    decision_file = Path(packet.get("decision_file") or path.with_suffix(".decision.json"))
    allowed = packet.get("allowed_actions") or []

    prompt = f"""
Project root: {PROJECT_ROOT}
Escalation packet path: {path}
Decision file path: {decision_file}
Allowed actions: {allowed}

Escalation packet:
```json
{json.dumps(packet, indent=2)[:60000]}
```

Read any referenced artifacts from disk. Then return this JSON shape:
{{
  "action": "one of the allowed actions",
  "rationale": "why this action is the right next step",
  "root_cause": "short root-cause classification",
  "evidence": ["file/path: concrete evidence", "..."],
  "confidence": 0.0,
  "block_actions": {{}},
  "human_escalation_needed": false
}}
"""

    if dry_run:
        return _fallback_decision(packet, "dry-run: no LLM triage executed")

    llm = ClaudeLLM(
        model=os.environ.get("CORESMITH_TRIAGE_MODEL", DEFAULT_MODEL),
        timeout=int(os.environ.get("CORESMITH_TRIAGE_TIMEOUT_S", "1200")),
    )
    text = await llm.call(system=SYSTEM, prompt=prompt, run_name="headless_triage")
    defaults = {
        "action": "abort" if "abort" in allowed else allowed[0] if allowed else "abort",
        "rationale": "triage output did not parse cleanly",
        "root_cause": "UNKNOWN",
        "evidence": [],
        "confidence": 0.0,
        "block_actions": {},
        "human_escalation_needed": True,
    }
    decision, parse_ok = parse_llm_json(text, defaults, context="headless triage")
    if not parse_ok and _is_transient_llm_failure(text):
        raise RuntimeError(
            "transient LLM triage failure; leaving decision file absent so "
            "the headless runner can retry triage"
        )
    if decision.get("action") not in allowed:
        decision = _fallback_decision(
            packet,
            f"triage chose unsupported action {decision.get('action')!r}; allowed={allowed}",
        )
    decision["triage_agent"] = {
        "packet": str(path),
        "model": os.environ.get("CORESMITH_TRIAGE_MODEL", DEFAULT_MODEL),
        "parse_ok": parse_ok,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(decision_file, json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--escalation", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    decision = asyncio.run(triage(Path(args.escalation), dry_run=args.dry_run))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
