# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architect sitting step 5: chip lead = the architect resumed."""
import json

from orchestrator.architect import consult as C
from orchestrator.state_store.project_db import open_project


class _FakeSession:
    def __init__(self, text, sid="sess-9"):
        self._text, self._sid, self.calls = text, sid, []
        from pathlib import Path
        self.dir = Path("/tmp")

    def sit(self, prompt, *, resume="", index=1):
        self.calls.append((prompt, resume, index))
        return {"ok": True, "text": self._text, "session_id": self._sid}


_PAYLOAD = {"type": "uarch_integration_review", "tier": 1, "supported_actions": ["approve", "revise", "abort"],
            "affected_blocks": ["alu", "ctl"], "findings": ["alu §6a says 2 cycles, ctl expects 1"]}


def test_consult_resumes_the_session_and_parses_the_decision(tmp_path):
    db = open_project(tmp_path)
    db.set_setting("architect_session_id", "sess-9")
    fake = _FakeSession('Thinking...\n```json\n{"action": "revise", "reasoning": "TIM-alu mismatch", '
                        '"feedback": "make ctl accept 2-cycle response", "block_actions": {"ctl": "revise"}}\n```')
    d = C.consult(tmp_path, _PAYLOAD, ["{\"action\": \"approve\"}"], session=fake)
    assert d["action"] == "revise" and d["block_actions"] == {"ctl": "revise"} and d["decided_by"] == "architect"
    prompt, resume, index = fake.calls[0]
    assert resume == "sess-9" and index == 1001 and "uarch_integration_review" in prompt and "prior decisions" in prompt
    assert (tmp_path / ".coresmith" / "architect" / "consult-1.md").exists()
    assert json.loads((tmp_path / ".coresmith" / "architect" / "consult-1.decision.json").read_text())["action"] == "revise"


def test_consult_falls_back_without_a_session_or_a_parseable_answer(tmp_path):
    assert C.consult(tmp_path, _PAYLOAD) is None                      # no session id recorded
    db = open_project(tmp_path)
    db.set_setting("architect_session_id", "sess-1")
    assert C.consult(tmp_path, _PAYLOAD, session=_FakeSession("I cannot decide.")) is None
    assert C.consult(tmp_path, _PAYLOAD, session=_FakeSession('```json\n{"reasoning": "no action"}\n```')) is None


def test_consult_prompt_forbids_abort_and_carries_the_schema():
    p = C.consult_prompt(_PAYLOAD, [])
    assert "`abort` is never yours" in p and '"action": "<one of the payload' in p and "supported_actions" in p
