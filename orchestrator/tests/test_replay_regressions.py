# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Replay regression tests (Package C, C4).

Each test replays a REAL recorded LLM behavior (extracted from an archived run
by ``scripts/make_replay_fixture.py``) and asserts the plumbing outcome the
engine must produce for that behavior -- prompt-content-independent, keyed on
``(run_name, site_index)``. These are permanent regression guards: if a refactor
reintroduces a fail-open, the recorded response reproduces it deterministically
in milliseconds instead of one-per-4h-run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from pathlib import Path

import pytest

from orchestrator.testing import replay_provider as rp
from orchestrator.testing.replay_provider import ReplayBackend, ReplayMissError
from orchestrator.testing.prompt_norm import (
    normalize_prompt,
    prompt_digest,
    token_jaccard,
)

pytestmark = pytest.mark.replay

_FIXTURES = Path(__file__).parent / "fixtures" / "replay"
_REPO = Path(__file__).resolve().parents[2]


def _set_site(run_name: str, call_index: int = 1) -> None:
    from orchestrator.langchain.agents import coresmith_llm as cl

    cl._call_site_context.set({"run_name": run_name, "call_index": call_index})


# ---------------------------------------------------------------------------
# prompt_norm unit coverage
# ---------------------------------------------------------------------------
class TestPromptNorm:
    def test_root_and_time_and_id_normalized(self):
        a = ("Project root: /home/ubuntu/coresmith-runs/job-20260513-071312\n"
             "at 2026-05-13T07:11:34 session tok_abc123DEF456 "
             "sha=deadbeefdeadbeefdeadbeef")
        n = normalize_prompt(a, roots=["/home/ubuntu/coresmith-runs/job-20260513-071312"])
        assert "<RUN>" in n and "2026-05-13" not in n and "tok_abc123DEF456" not in n
        assert "<T>" in n and "<ID>" in n

    def test_digest_stable_across_roots(self):
        base = "write RTL to {root}/rtl/x.v now"
        d1 = prompt_digest(base.format(root="/home/ubuntu/project"),
                           roots=["/home/ubuntu/project"])
        d2 = prompt_digest(base.format(root="/tmp/pytest/t0"),
                           roots=["/tmp/pytest/t0"])
        assert d1 == d2

    def test_jaccard_bounds(self):
        assert token_jaccard("a b c", "a b c") == 1.0
        assert token_jaccard("", "") == 1.0
        assert token_jaccard("a b c", "") == 0.0
        assert 0.0 < token_jaccard("alpha beta gamma", "alpha beta delta") < 1.0


# ---------------------------------------------------------------------------
# Fixture 1: Integration Lead JSON-vs-disk mismatch -> RAISE (fail-closed)
# ---------------------------------------------------------------------------
class TestIntegrationLeadJsonDiskMismatch:
    """The recorded response is ``{"verilog": "<prose: written to disk>"}`` with
    no on-disk module. The plumbing guard (integration_lead._has_real_module) must
    refuse to write a recursive-include stub and RAISE so the node retries."""

    def test_raises_on_prose_verilog_no_disk_file(self, replay_llm, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        replay_llm("integration_lead_json_disk_mismatch", strict=True)

        from orchestrator.langchain.agents.integration_lead import IntegrationLeadAgent

        agent = IntegrationLeadAgent()
        out_path = tmp_path / "rtl" / "integration" / "chip_top.v"
        with pytest.raises(RuntimeError) as ei:
            asyncio.run(agent.integrate(
                design_name="chip_top",
                block_rtl_sources={},
                block_port_summaries=[],
                connections=[],
                prd_summary="",
                output_path=str(out_path),
            ))
        assert "no valid top-level Verilog module" in str(ei.value)
        # Fail-closed: it must NOT have written a bogus top.
        assert not out_path.exists()

    def test_recorded_response_is_the_real_mismatch_shape(self):
        """Guard the fixture itself: response is prose-in-verilog, not a module."""
        b = ReplayBackend(_FIXTURES / "integration_lead_json_disk_mismatch")
        _set_site("Integration Lead [chip_top]")
        resp = b.generate(None, "sys", "prompt", None)
        data = json.loads(resp[resp.index("{"):resp.rindex("}") + 1])
        verilog = data["verilog"]
        assert "written to" in verilog  # prose, not code
        assert len(verilog.strip().splitlines()) < 5  # the guard's rejection key


# ---------------------------------------------------------------------------
# Fixture 2: uarch-spec summary-only response (agent wrote artifact via tools)
# ---------------------------------------------------------------------------
class TestUarchSpecSummaryReplay:
    def test_exact_match_serves_recorded_response(self):
        b = ReplayBackend(_FIXTURES / "uarch_spec_summary_response", strict=True)
        _set_site("Generate Uarch Spec [Intra Predict]")
        resp = b.generate(None, "sys", "any prompt", None)
        # The response is a markdown-link SUMMARY -- it carries no RTL/artifact.
        assert "arch/uarch_specs/intra_predict.md" in resp
        assert "module " not in resp
        assert b.call_log[-1]["matched"] == "exact"
        assert not b.unconsumed  # consumed

    def test_strict_miss_raises_with_diagnostics(self):
        b = ReplayBackend(_FIXTURES / "uarch_spec_summary_response", strict=True)
        _set_site("Generate Uarch Spec [Nonexistent Block]")
        with pytest.raises(ReplayMissError) as ei:
            b.generate(None, "sys", "prompt", None)
        msg = str(ei.value)
        assert "no recorded call matched" in msg
        assert "Nonexistent Block" in msg

    def test_lenient_miss_falls_back_to_canned(self):
        b = ReplayBackend(_FIXTURES / "uarch_spec_summary_response", strict=False)
        _set_site("Generate Uarch Spec [Nonexistent Block]")
        resp = b.generate(None, "sys", "no target paths here", None)
        assert "Done" in resp  # CannedDesignScript response, no raise


# ---------------------------------------------------------------------------
# Write-application + --auto-writes (synthetic run, deterministic)
# ---------------------------------------------------------------------------
def _load_make_fixture_module():
    path = _REPO / "scripts" / "make_replay_fixture.py"
    spec = importlib.util.spec_from_file_location("make_replay_fixture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAutoWritesAndApplication:
    def _synth_run(self, run_dir: Path) -> None:
        """Build a minimal recorded run: one llm call that 'wrote' spec.md."""
        cor = run_dir / ".coresmith"
        cor.mkdir(parents=True)
        t0 = time.time() - 100.0
        # llm_start ... (file mtime here) ... llm_end
        events = [
            {"event": "llm_start", "ts": t0, "run_name": "Generate Uarch Spec [foo]",
             "system_chars": 3, "prompt_chars": 5},
            {"event": "llm_end", "ts": t0 + 20.0, "run_name": "Generate Uarch Spec [foo]",
             "output_chars": 4},
        ]
        (cor / "pipeline_events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n")
        (cor / "llm_calls.jsonl").write_text(json.dumps({
            "ts": t0 + 20.0, "iso": "2026-05-13T07:00:00", "provider": "codex_cli",
            "model": "gpt-5.5", "system_prompt": "sys", "user_prompt": "hello",
            "response": "wrote", "response_len": 4, "system_prompt_len": 3,
            "user_prompt_len": 5, "duration_s": 20.0, "timeout": 2700,
            "timed_out": False, "error": "", "usage": {},
        }) + "\n")
        # The artifact the call "wrote", mtime inside the window.
        art = run_dir / "arch" / "spec.md"
        art.parent.mkdir(parents=True)
        art.write_text("# spec for foo\n")
        import os as _os
        _os.utime(art, (t0 + 5.0, t0 + 5.0))

    def test_auto_writes_captures_in_window_file_and_replay_applies_it(
            self, tmp_path, monkeypatch):
        mod = _load_make_fixture_module()
        run_dir = tmp_path / "recorded_run"
        run_dir.mkdir()
        self._synth_run(run_dir)

        out = tmp_path / "fx"
        meta = mod.build_fixture(
            source=str(run_dir), out=str(out), name="synthwrite",
            indices=[0], infer_sites=True, auto_writes=True,
        )
        assert meta["total_writes"] == 1
        rec = json.loads((out / "calls.jsonl").read_text().splitlines()[0])
        assert rec["writes"][0]["relpath"] == "arch/spec.md"
        # Telemetry files must NOT be captured as writes.
        assert all("llm_calls" not in w["relpath"] for w in rec["writes"])

        # Replay under a DIFFERENT project root -> the write lands there.
        new_root = tmp_path / "replay_root"
        new_root.mkdir()
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(new_root))
        b = ReplayBackend(out, strict=True)
        _set_site("Generate Uarch Spec [foo]")
        resp = b.generate(None, "sys", "hello", None)
        assert resp == "wrote"
        applied = new_root / "arch" / "spec.md"
        assert applied.exists() and applied.read_text() == "# spec for foo\n"


# ---------------------------------------------------------------------------
# Matching cascade (digest / fuzzy)
# ---------------------------------------------------------------------------
class TestMatchCascade:
    _CANON = ("generate the synthesizable verilog module for block foo with "
              "clk rst valid ready and data ports per the spec")

    def _mk_fixture(self, root: Path, *, keep_prompts: bool) -> Path:
        (root / "files").mkdir(parents=True)
        resp_ref = "files/r0"
        (root / "files" / "r0").write_text("RESP0")
        rec = {
            "run_name": "N", "site_index": 0, "provider": "p", "model": "m",
            "prompt_digest": prompt_digest(self._CANON, []),
            "response_ref": resp_ref, "response_len": 5, "writes": [],
        }
        if keep_prompts:
            rec["prompt_norm"] = normalize_prompt(self._CANON, [])
        (root / "calls.jsonl").write_text(json.dumps(rec) + "\n")
        (root / "meta.json").write_text(json.dumps({"original_root": ""}))
        return root

    def test_digest_match_when_site_index_off(self, tmp_path):
        fx = self._mk_fixture(tmp_path / "fx", keep_prompts=False)
        b = ReplayBackend(fx, strict=True)
        # Pre-serve advances the served counter so site_index 0 is 'past'.
        b._served["N"] = 5
        _set_site("N")
        resp = b.generate(None, "sys", self._CANON, None)
        assert resp == "RESP0"
        assert b.call_log[-1]["matched"] == "digest"

    def test_fuzzy_match_needs_kept_prompt(self, tmp_path):
        fx = self._mk_fixture(tmp_path / "fx", keep_prompts=True)
        b = ReplayBackend(fx, strict=True)
        b._served["N"] = 5  # skip exact
        _set_site("N")
        # Near-identical prompt (one word changed, ~0.87 jaccard) -> digest
        # differs, fuzzy hits above the 0.80 default threshold.
        near = self._CANON.replace("data", "payload")
        resp = b.generate(None, "sys", near, None)
        assert resp == "RESP0"
        assert b.call_log[-1]["matched"].startswith("fuzzy")
