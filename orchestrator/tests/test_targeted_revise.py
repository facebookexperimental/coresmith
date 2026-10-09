"""WP-7: targeted integration-review revise + single-context uArch stage.

Before: a chip-level ``revise`` re-fanned out the WHOLE tier from spec
generation, the reviewer's edits (made on ``arch/uarch_specs_review/`` copies)
were never adopted, and the chip lead's feedback never reached the spec
author. The h264 Arm E run spent 6 tier-3 rounds in that loop.

After: ``integration_review_node`` turns a revise into ``revise_blocks``
(``{block: reuse_spec}``): reviewer-edited specs are adopted as canonical and
those blocks implement them as-is, chip-lead-named blocks re-spec with the
findings as gate feedback, everything else keeps its passing result.
``CORESMITH_UARCH_SINGLE_CONTEXT=1`` authors all specs in one session at the
first tier entry instead of one author per block.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.langgraph import pipeline_graph, pipeline_helpers


def _seed_project(root: Path, names: list[str], tier: int = 1) -> None:
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / "arch" / "uarch_specs").mkdir(parents=True, exist_ok=True)
    bd = {
        "blocks": [{"name": n, "tier": tier, "interfaces": {}} for n in names],
        "connections": [],
    }
    (root / ".coresmith" / "block_diagram.json").write_text(json.dumps(bd))


def _orch_state(root: Path, names: list[str], **extra) -> dict:
    st = {
        "project_root": str(root),
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": [{"name": n, "tier": 1} for n in names],
        "tier_list": [1],
        "current_tier_index": 0,
        "completed_blocks": [],
    }
    st.update(extra)
    return st


def _patch_review(monkeypatch, result: dict):
    from orchestrator.langchain.agents import integration_review_agent as ira

    async def fake_review(self, block_names, project_root):
        return result

    monkeypatch.setattr(ira.IntegrationReviewAgent, "__init__",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(ira.IntegrationReviewAgent, "review", fake_review)


# --------------------------------------------------------------------------
# Review agent reports which blocks it edited
# --------------------------------------------------------------------------
class TestReviewAgentReportsEdits:
    @pytest.mark.asyncio
    async def test_edited_blocks_and_reviewed_paths(self, tmp_path, monkeypatch):
        from orchestrator.langchain.agents import integration_review_agent as ira

        _seed_project(tmp_path, ["alpha", "beta"])
        for n in ("alpha", "beta"):
            (tmp_path / "arch" / "uarch_specs" / f"{n}.md").write_text(f"# {n}\n")

        class FakeLLM:
            async def call(self, system="", prompt="", run_name="", **kw):
                # the reviewer edits ONE review copy
                p = tmp_path / "arch" / "uarch_specs_review" / "alpha.md"
                p.write_text("# alpha (renamed port)\n")
                return 'fixed one\n```json\n{"issues_found": 1, "issues_fixed": 1}\n```'

        monkeypatch.delenv("CORESMITH_INTEGRATION_REVIEW_INPLACE", raising=False)
        agent = ira.IntegrationReviewAgent.__new__(ira.IntegrationReviewAgent)
        agent.llm = FakeLLM()
        out = await agent.review(block_names=["alpha", "beta"],
                                 project_root=str(tmp_path))
        assert out["edited_blocks"] == ["alpha"]
        assert Path(out["reviewed_specs"]["alpha"]).read_text().startswith("# alpha (renamed")
        assert "beta" in out["reviewed_specs"]
        # canonical spec untouched by the review itself (adoption is the node's job)
        assert (tmp_path / "arch" / "uarch_specs" / "alpha.md").read_text() == "# alpha\n"


# --------------------------------------------------------------------------
# integration_review_node: targeted plan
# --------------------------------------------------------------------------
class TestTargetedRevisePlan:
    @pytest.mark.asyncio
    async def test_revise_scopes_to_edited_plus_named_blocks(self, tmp_path, monkeypatch):
        names = ["alpha", "beta", "gamma"]
        _seed_project(tmp_path, names)
        specs = tmp_path / "arch" / "uarch_specs"
        for n in names:
            (specs / f"{n}.md").write_text(f"# {n} v1\n")
        review_copy = tmp_path / "arch" / "uarch_specs_review" / "alpha.md"
        review_copy.parent.mkdir(parents=True)
        review_copy.write_text("# alpha v2 (reviewer edit)\n")
        _patch_review(monkeypatch, {
            "summary": "renamed alpha ports",
            "issues_found": 1, "issues_fixed": 1,
            "edited_blocks": ["alpha"],
            "reviewed_specs": {n: str(review_copy if n == "alpha" else specs / f"{n}.md")
                               for n in names},
        })
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {
            "action": "revise",
            "affected_blocks": ["beta"],
            "feedback": "beta must expose m_token_srdy/m_token_drdy",
        })
        monkeypatch.setenv("CORESMITH_STRICT_INTEGRATION_REVIEW", "0")
        out = await pipeline_graph.integration_review_node(_orch_state(tmp_path, names))

        assert out["integration_review_action"] == "revise"
        assert out["revise_blocks"] == {"alpha": True, "beta": False}
        # reviewer edit adopted as canonical
        assert (specs / "alpha.md").read_text() == "# alpha v2 (reviewer edit)\n"
        # named block gets the chip lead's finding as gate feedback
        fb = tmp_path / ".coresmith" / "blocks" / "beta" / "gate_feedback.txt"
        assert "m_token_srdy" in fb.read_text()
        assert not (tmp_path / ".coresmith" / "blocks" / "alpha" / "gate_feedback.txt").exists()
        # untouched block is not in the plan and keeps its spec
        assert (specs / "gamma.md").read_text() == "# gamma v1\n"
        assert not (tmp_path / ".coresmith" / "blocks" / "gamma").exists()

    @pytest.mark.asyncio
    async def test_unscoped_revise_reenters_whole_tier_with_review_feedback(self, tmp_path, monkeypatch):
        names = ["alpha", "beta"]
        _seed_project(tmp_path, names)
        for n in names:
            (tmp_path / "arch" / "uarch_specs" / f"{n}.md").write_text(f"# {n}\n")
        _patch_review(monkeypatch, {
            "summary": "widths disagree on edge e1", "issues_found": 2,
            "issues_fixed": 0, "edited_blocks": [], "reviewed_specs": {},
        })
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda payload: {"action": "revise"})
        out = await pipeline_graph.integration_review_node(_orch_state(tmp_path, names))
        assert out["revise_blocks"] == {"alpha": False, "beta": False}
        for n in names:
            fb = tmp_path / ".coresmith" / "blocks" / n / "gate_feedback.txt"
            assert "widths disagree on edge e1" in fb.read_text()

    @pytest.mark.asyncio
    async def test_named_by_mention_in_reasoning(self, tmp_path, monkeypatch):
        # the old prose-mention rule, behind CORESMITH_REVISE_PROSE_TARGETS=1
        # (default off: see test_mcufft_run3_fixes.py::test_prose_never_targets)
        monkeypatch.setenv("CORESMITH_REVISE_PROSE_TARGETS", "1")
        names = ["alpha", "beta"]
        _seed_project(tmp_path, names)
        for n in names:
            (tmp_path / "arch" / "uarch_specs" / f"{n}.md").write_text(f"# {n}\n")
        _patch_review(monkeypatch, {
            "summary": "ok", "issues_found": 1, "issues_fixed": 0,
            "edited_blocks": [], "reviewed_specs": {},
        })
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {
            "action": "revise",
            "reasoning": "beta still lacks the packed handshake; alphabet is fine",
        })
        out = await pipeline_graph.integration_review_node(_orch_state(tmp_path, names))
        # exact-name mention only: 'alphabet' does not name 'alpha'
        assert out["revise_blocks"] == {"beta": False}

    @pytest.mark.asyncio
    async def test_approve_missing_reviewed_source_parks_and_lists_edits(self, tmp_path, monkeypatch):
        names = ["alpha"]
        _seed_project(tmp_path, names)
        (tmp_path / "arch" / "uarch_specs" / "alpha.md").write_text("# alpha\n")
        _patch_review(monkeypatch, {
            "summary": "ok", "issues_found": 1, "issues_fixed": 1,
            "edited_blocks": ["alpha"], "reviewed_specs": {},
        })
        seen = {}

        def fake_interrupt(payload):
            seen.update(payload)
            return {"action": "approve"}

        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)
        monkeypatch.delenv("CORESMITH_STRICT_INTEGRATION_REVIEW", raising=False)
        out = await pipeline_graph.integration_review_node(_orch_state(tmp_path, names))
        assert seen["edited_blocks"] == ["alpha"]
        assert out["integration_review_action"] == "abort"
        assert out["integration_review_failed"]
        assert pipeline_graph.route_after_integration_review(out) != "advance_tier"

    def test_revise_named_blocks_accepts_block_actions_json(self):
        resp = {"block_actions": json.dumps({"beta": "restart", "alpha": "approve"})}
        assert pipeline_graph._revise_named_blocks(resp, ["alpha", "beta", "gamma"]) == ["beta"]
        assert pipeline_graph._revise_named_blocks(
            {"affected_blocks": "gamma, alpha"}, ["alpha", "beta", "gamma"]) == ["alpha", "gamma"]


# --------------------------------------------------------------------------
# fan-out / tier advance honour the plan
# --------------------------------------------------------------------------
class TestFanOutHonoursPlan:
    def test_only_planned_blocks_are_sent_with_reuse_flag(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_UARCH_SINGLE_CONTEXT", raising=False)
        st = _orch_state(tmp_path, ["alpha", "beta", "gamma"],
                         revise_blocks={"alpha": True, "beta": False})
        sends = pipeline_graph.fan_out_tier(st)
        got = {s.arg["current_block"]["name"]: s.arg["reuse_spec"] for s in sends}
        assert got == {"alpha": True, "beta": False}

    def test_normal_entry_sends_every_block_without_reuse(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_UARCH_SINGLE_CONTEXT", raising=False)
        monkeypatch.setenv("CORESMITH_UARCH_PHASE", "0")   # B2: the phase also implies spec reuse
        sends = pipeline_graph.fan_out_tier(_orch_state(tmp_path, ["alpha", "beta"]))
        assert [s.arg["reuse_spec"] for s in sends] == [False, False]

    def test_single_context_mode_sends_reuse(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_UARCH_SINGLE_CONTEXT", "1")
        sends = pipeline_graph.fan_out_tier(_orch_state(tmp_path, ["alpha", "beta"]))
        assert [s.arg["reuse_spec"] for s in sends] == [True, True]

    @pytest.mark.asyncio
    async def test_advance_tier_clears_plan(self, tmp_path):
        out = await pipeline_graph.advance_tier_node(
            _orch_state(tmp_path, ["alpha"], revise_blocks={"alpha": True}))
        assert out["revise_blocks"] is None
        assert out["current_tier_index"] == 1

    def test_route_still_reenters_init_tier(self):
        assert pipeline_graph.route_after_integration_review(
            {"integration_review_action": "revise"}) == "init_tier"


# --------------------------------------------------------------------------
# generate_uarch_spec_node reuse branch
# --------------------------------------------------------------------------
def _block_state(root: Path, name: str, **extra) -> dict:
    st = {
        "project_root": str(root),
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "pipeline_run_start": 0.0,
        "current_block": {"name": name, "tier": 1, "description": "d"},
        "attempt": 1, "phase": "init", "human_response": None,
        "reuse_spec": True,
    }
    st.update(extra)
    return st


class TestGenerateUarchSpecReuse:
    @pytest.mark.asyncio
    async def test_reuses_on_disk_spec_without_calling_the_author(self, tmp_path, monkeypatch):
        _seed_project(tmp_path, ["alpha"])
        (tmp_path / "arch" / "uarch_specs" / "alpha.md").write_text("# alpha reviewed\n")

        async def boom(*a, **k):
            raise AssertionError("spec author must not be called")

        monkeypatch.setattr(pipeline_graph, "generate_uarch_spec", boom)
        monkeypatch.delenv("CORESMITH_IGNORE_SPEC_PINS", raising=False)
        out = await pipeline_graph.generate_uarch_spec_node(_block_state(tmp_path, "alpha"))
        assert out == {"uarch_approved": False, "phase": "uarch"}
        assert (tmp_path / "arch" / "uarch_specs" / "alpha.md").read_text() == "# alpha reviewed\n"

    @pytest.mark.asyncio
    async def test_pending_gate_feedback_still_revises_per_block(self, tmp_path, monkeypatch):
        _seed_project(tmp_path, ["alpha"])
        (tmp_path / "arch" / "uarch_specs" / "alpha.md").write_text("# alpha v1\n")
        bdir = tmp_path / ".coresmith" / "blocks" / "alpha"
        bdir.mkdir(parents=True)
        (bdir / "gate_feedback.txt").write_text("add m_token_srdy")
        calls = []

        async def fake_generate(block, feedback="", previous_spec="", **k):
            calls.append((feedback, previous_spec))
            return {"spec_text": "# alpha v2\n", "block_name": block["name"]}

        monkeypatch.setattr(pipeline_graph, "generate_uarch_spec", fake_generate)
        out = await pipeline_graph.generate_uarch_spec_node(_block_state(tmp_path, "alpha"))
        assert out["phase"] == "uarch"
        assert len(calls) == 1
        assert "add m_token_srdy" in calls[0][0]
        assert calls[0][1] == "# alpha v1\n"

    @pytest.mark.asyncio
    async def test_missing_spec_falls_through_to_the_author(self, tmp_path, monkeypatch):
        _seed_project(tmp_path, ["alpha"])
        calls = []

        async def fake_generate(block, **k):
            calls.append(block["name"])
            return {"spec_text": "# alpha\n", "block_name": block["name"]}

        monkeypatch.setattr(pipeline_graph, "generate_uarch_spec", fake_generate)
        await pipeline_graph.generate_uarch_spec_node(_block_state(tmp_path, "alpha"))
        assert calls == ["alpha"]


# --------------------------------------------------------------------------
# single-context uArch stage in init_tier
# --------------------------------------------------------------------------
class TestSingleContextStage:
    @pytest.mark.asyncio
    async def test_first_entry_authors_every_missing_spec_once(self, tmp_path, monkeypatch):
        names = ["alpha", "beta", "gamma"]
        _seed_project(tmp_path, names)
        (tmp_path / ".coresmith" / "block_specs.json").write_text(json.dumps(
            {"blocks": [{"name": n, "tier": 1} for n in names]}))
        monkeypatch.setenv("CORESMITH_UARCH_SINGLE_CONTEXT", "1")
        calls = []

        async def fake_single(blocks, feedback_by_block=None):
            calls.append(([b["name"] for b in blocks], dict(feedback_by_block or {})))
            for b in blocks:
                (tmp_path / "arch" / "uarch_specs" / f"{b['name']}.md").write_text("# spec\n")
            return {"written": [b["name"] for b in blocks], "missing": []}

        monkeypatch.setattr(pipeline_helpers, "generate_uarch_specs_single_context", fake_single)
        st = _orch_state(tmp_path, names)
        out = await pipeline_graph.init_tier_node(st)
        assert calls == [(names, {})]
        assert "revise_blocks" not in out
        # second entry (next tier / re-entry): nothing missing, no call
        st.update(out)
        await pipeline_graph.init_tier_node(st)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_revise_reentry_respecs_named_blocks_in_one_session(self, tmp_path, monkeypatch):
        names = ["alpha", "beta"]
        _seed_project(tmp_path, names)
        for n in names:
            (tmp_path / "arch" / "uarch_specs" / f"{n}.md").write_text(f"# {n}\n")
        bdir = tmp_path / ".coresmith" / "blocks" / "beta"
        bdir.mkdir(parents=True)
        (bdir / "gate_feedback.txt").write_text("rename ports")
        monkeypatch.setenv("CORESMITH_UARCH_SINGLE_CONTEXT", "1")
        calls = []

        async def fake_single(blocks, feedback_by_block=None):
            calls.append(([b["name"] for b in blocks], dict(feedback_by_block or {})))
            return {"written": ["beta"], "missing": []}

        monkeypatch.setattr(pipeline_helpers, "generate_uarch_specs_single_context", fake_single)
        st = _orch_state(tmp_path, names, revise_blocks={"alpha": True, "beta": False})
        out = await pipeline_graph.init_tier_node(st)
        assert calls == [(["beta"], {"beta": "rename ports"})]
        assert out["revise_blocks"] == {"alpha": True, "beta": True}
        assert not (bdir / "gate_feedback.txt").exists()

    @pytest.mark.asyncio
    async def test_author_failure_falls_back_to_per_block(self, tmp_path, monkeypatch):
        names = ["alpha"]
        _seed_project(tmp_path, names)
        monkeypatch.setenv("CORESMITH_UARCH_SINGLE_CONTEXT", "1")

        async def fake_single(blocks, feedback_by_block=None):
            raise RuntimeError("codex down")

        monkeypatch.setattr(pipeline_helpers, "generate_uarch_specs_single_context", fake_single)
        out = await pipeline_graph.init_tier_node(_orch_state(tmp_path, names))
        assert out["tier_list"] == [1]
        assert "revise_blocks" not in out

    @pytest.mark.asyncio
    async def test_helper_reports_written_vs_missing(self, tmp_path, monkeypatch):
        from orchestrator.langchain.agents import uarch_spec_generator as usg

        _seed_project(tmp_path, ["alpha", "beta"])
        monkeypatch.setattr(pipeline_helpers, "PROJECT_ROOT", tmp_path)
        body = "# alpha\n\n## Interface\n" + ("| port | w |\n" * 60)

        async def fake_many(self, blocks, **kw):
            (tmp_path / "arch" / "uarch_specs" / "alpha.md").write_text(body)
            return "done"

        monkeypatch.setattr(usg.UarchSpecGenerator, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(usg.UarchSpecGenerator, "generate_many", fake_many)
        out = await pipeline_helpers.generate_uarch_specs_single_context(
            [{"name": "alpha"}, {"name": "beta"}])
        assert out["written"] == ["alpha"]
        assert out["missing"] == ["beta"]

    @pytest.mark.asyncio
    async def test_generate_many_prompt_names_every_block_and_contract(self, tmp_path):
        from orchestrator.langchain.agents import uarch_spec_generator as usg

        _seed_project(tmp_path, ["alpha", "beta"])
        (tmp_path / ".coresmith" / "interface_contracts.json").write_text(json.dumps({
            "contracts": [{"edge_id": "e1", "producer_block": "alpha",
                           "consumer_block": "beta", "signal": "m_token"}],
        }))
        (tmp_path / "arch" / "ers_spec.md").write_text("ERS TEXT")
        seen = {}

        class FakeLLM:
            async def call(self, system="", prompt="", run_name="", **kw):
                seen.update(system=system, prompt=prompt, run_name=run_name)
                return "ok"

        agent = usg.UarchSpecGenerator.__new__(usg.UarchSpecGenerator)
        agent.llm = FakeLLM()
        await agent.generate_many(
            blocks=[{"name": "alpha", "description": "A"}, {"name": "beta", "description": "B"}],
            python_sources={"alpha": "def f(): pass"},
            feedback={"beta": "rename ports"},
            previous_specs={"beta": "# beta old"},
            project_root=str(tmp_path),
        )
        p = seen["prompt"]
        assert "arch/uarch_specs/alpha.md" in p and "arch/uarch_specs/beta.md" in p
        assert "ERS TEXT" in p and '"edge_id": "e1"' in p
        assert "def f(): pass" in p
        assert "rename ports" in p and "REVISION REQUESTED" in p
        assert seen["run_name"].startswith("Generate Uarch Specs [2 blocks")


# --------------------------------------------------------------------------
# WP-7b: DV-failure revise plans span tiers
# --------------------------------------------------------------------------
class TestPlanSpansTiers:
    def _queue(self):
        return [{"name": "mem", "tier": 1}, {"name": "ctl", "tier": 2},
                {"name": "enc", "tier": 3}]

    @pytest.mark.asyncio
    async def test_init_tier_skips_tiers_with_nothing_to_redo(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_UARCH_SINGLE_CONTEXT", raising=False)
        _seed_project(tmp_path, ["mem", "ctl", "enc"])
        st = _orch_state(tmp_path, [], block_queue=self._queue(), tier_list=[1, 2, 3],
                         current_tier_index=0, revise_blocks={"enc": False})
        out = await pipeline_graph.init_tier_node(st)
        assert out["current_tier_index"] == 2
        st.update(out)
        sends = pipeline_graph.fan_out_tier(st)
        assert [s.arg["current_block"]["name"] for s in sends] == ["enc"]
        assert sends[0].arg["reuse_spec"] is False

    @pytest.mark.asyncio
    async def test_plan_naming_no_queued_block_voids_itself(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_UARCH_SINGLE_CONTEXT", raising=False)
        _seed_project(tmp_path, ["mem"])
        st = _orch_state(tmp_path, [], block_queue=self._queue(), tier_list=[1, 2, 3],
                         revise_blocks={"ghost": False})
        out = await pipeline_graph.init_tier_node(st)
        assert out["revise_blocks"] is None
        assert "current_tier_index" not in out
        st.update(out)
        assert len(pipeline_graph.fan_out_tier(st)) == 1  # tier 1 runs normally

    @pytest.mark.asyncio
    async def test_advance_tier_keeps_plan_while_later_tiers_are_planned(self, tmp_path):
        st = _orch_state(tmp_path, [], block_queue=self._queue(), tier_list=[1, 2, 3],
                         current_tier_index=0, revise_blocks={"mem": False, "enc": False})
        out = await pipeline_graph.advance_tier_node(st)
        assert out == {"current_tier_index": 1, "revise_blocks": {"mem": False, "enc": False}}
        st["current_tier_index"] = 2
        out = await pipeline_graph.advance_tier_node(st)
        assert out["revise_blocks"] is None

    def test_apply_revise_uarch_hands_feedback_to_the_block(self, tmp_path):
        _seed_project(tmp_path, ["enc"])
        (tmp_path / "arch" / "uarch_specs" / "enc.md").write_text("# enc\n")
        applied = pipeline_graph._apply_revise_uarch(
            str(tmp_path), {"action": "revise", "affected_blocks": ["enc"],
                            "feedback": "token FIFO must be shared"},
            {}, "validation_dv")
        assert applied == ["enc"]
        fb = (tmp_path / ".coresmith" / "blocks" / "enc" / "gate_feedback.txt").read_text()
        assert "token FIFO must be shared" in fb
        assert "VALIDATION_DV REVISION FEEDBACK" in (tmp_path / "arch" / "uarch_specs" / "enc.md").read_text()

    @pytest.mark.asyncio
    async def test_integration_review_carries_later_tier_entries(self, tmp_path, monkeypatch):
        _seed_project(tmp_path, ["mem", "enc"])
        (tmp_path / "arch" / "uarch_specs" / "mem.md").write_text("# mem\n")
        _patch_review(monkeypatch, {
            "summary": "ok", "issues_found": 0, "issues_fixed": 0,
            "edited_blocks": [], "reviewed_specs": {},
        })
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "approve"})
        st = _orch_state(tmp_path, [], block_queue=self._queue(), tier_list=[1, 2, 3],
                         current_tier_index=0, revise_blocks={"mem": False, "enc": False})
        out = await pipeline_graph.integration_review_node(st)
        assert out["integration_review_action"] == "approve"
        assert out["revise_blocks"] == {"enc": False}


class TestContractPortGateHandshake:
    EDGE = {
        "edge_id": "intra__m_residual__to__forward__s_residual",
        "producer_block": "intra16", "producer_port": "m_residual_srdy/m_residual_data",
        "consumer_block": "forward", "consumer_port": "s_residual_drdy/s_residual_data",
        "handshake_protocol": "srdy_drdy", "data_width_bits": 12,
        "fields": [{"name": "samples", "width": 9}, {"name": "block_class", "width": 3}],
    }

    def _project(self, tmp_path, rtl: str):
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "interface_contracts.json").write_text(
            json.dumps({"contracts": [self.EDGE]}))
        p = tmp_path / "forward.v"
        p.write_text(rtl)
        return p

    def test_flattened_fields_with_handshake_pass(self, tmp_path):
        p = self._project(tmp_path, """module forward (
    input wire clk, input wire rst_n,
    input wire s_residual_srdy, output wire s_residual_drdy,
    input wire [8:0] s_residual_samples,
    input wire [2:0] s_residual_block_class
);
endmodule
""")
        assert pipeline_helpers.check_rtl_contract_ports(tmp_path, "forward", str(p)) == []

    def test_wrong_field_width_is_reported(self, tmp_path):
        p = self._project(tmp_path, """module forward (
    input wire clk, input wire rst_n,
    input wire s_residual_srdy, output wire s_residual_drdy,
    input wire [7:0] s_residual_samples,
    input wire [2:0] s_residual_block_class
);
endmodule
""")
        errs = pipeline_helpers.check_rtl_contract_ports(tmp_path, "forward", str(p))
        assert any("s_residual_samples" in e and "9 bits" in e for e in errs), errs

    def test_producer_side_pair_direction_agnostic(self, tmp_path):
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "interface_contracts.json").write_text(
            json.dumps({"contracts": [self.EDGE]}))
        p = tmp_path / "intra.v"
        p.write_text("""module intra16 (
    input wire clk, input wire rst_n,
    output wire m_residual_srdy, input wire m_residual_drdy,
    output wire [8:0] m_residual_samples, output wire [2:0] m_residual_block_class
);
endmodule
""")
        assert pipeline_helpers.check_rtl_contract_ports(tmp_path, "intra16", str(p)) == []


class TestSignalSpecsHandshake:
    def test_srdy_drdy_edge_yields_the_pair_once(self):
        from orchestrator.langgraph import contract_conformance as cc
        edge = {"handshake_protocol": "srdy_drdy",
                "fields": [{"name": "samples", "width": 9}],
                "sideband_signals": []}
        names = [s["name"] for s in cc.signal_specs(edge)]
        assert names == ["samples", "srdy", "drdy"]
        rows = cc.channel_signals(edge, "s_residual_drdy/s_residual_data")
        assert {r["port"] for r in rows} >= {"s_residual_samples", "s_residual_srdy", "s_residual_drdy"}

    def test_explicit_sideband_pair_not_duplicated(self):
        from orchestrator.langgraph import contract_conformance as cc
        edge = {"handshake_protocol": "srdy_drdy", "fields": [{"name": "v", "width": 4}],
                "sideband_signals": ["srdy", "drdy"]}
        names = [s["name"] for s in cc.signal_specs(edge)]
        assert names.count("srdy") == 1 and names.count("drdy") == 1

    def test_other_protocols_untouched(self):
        from orchestrator.langgraph import contract_conformance as cc
        edge = {"handshake_protocol": "req_resp", "fields": [{"name": "addr", "width": 8}],
                "sideband_signals": ["ren", "rvalid"]}
        assert [s["name"] for s in cc.signal_specs(edge)] == ["addr", "ren", "rvalid"]


class TestReviewFixes:
    """WP-11: fixes from the Codex review."""

    @pytest.mark.asyncio
    async def test_advance_tier_skips_finished_unplanned_tiers(self, tmp_path):
        queue = [{"name": "mem", "tier": 1}, {"name": "ctl", "tier": 2}, {"name": "enc", "tier": 3}]
        done = [{"name": "mem", "success": True}, {"name": "ctl", "success": True},
                {"name": "enc", "success": True}]
        # tier 1 just re-ran (targeted); tiers 2 and 3 already passed -> skip to the end
        out = await pipeline_graph.advance_tier_node(_orch_state(
            tmp_path, [], block_queue=queue, tier_list=[1, 2, 3], current_tier_index=0,
            completed_blocks=done, revise_blocks=None))
        assert out["current_tier_index"] == 3
        # first pass: nothing completed -> normal advance
        out = await pipeline_graph.advance_tier_node(_orch_state(
            tmp_path, [], block_queue=queue, tier_list=[1, 2, 3], current_tier_index=0))
        assert out["current_tier_index"] == 1
        # a plan naming a later block keeps that tier
        out = await pipeline_graph.advance_tier_node(_orch_state(
            tmp_path, [], block_queue=queue, tier_list=[1, 2, 3], current_tier_index=0,
            completed_blocks=done, revise_blocks={"enc": False}))
        assert out["current_tier_index"] == 2 and out["revise_blocks"] == {"enc": False}

    def test_measured_timing_failure_still_routes_to_diagnose(self):
        base = {"synth_success": True, "gate_sim_ok": None, "ppa_ok": False}
        assert pipeline_graph.route_after_synth({**base, "timing_ok": None}) == "evaluate_targets"
        assert pipeline_graph.route_after_synth({**base, "timing_ok": True}) == "evaluate_targets"
        assert pipeline_graph.route_after_synth({**base, "timing_ok": False}) == "diagnose"

    def test_explicit_keep_beats_prose_mention(self):
        resp = {"block_actions": {"alpha": "keep", "beta": "revise"},
                "feedback": "beta needs correction; alpha is correct and must stay unchanged"}
        assert pipeline_graph._revise_named_blocks(resp, ["alpha", "beta"]) == ["beta"]

    @pytest.mark.asyncio
    async def test_malformed_single_context_spec_is_quarantined(self, tmp_path, monkeypatch):
        from orchestrator.langchain.agents import uarch_spec_generator as usg
        _seed_project(tmp_path, ["alpha"])
        monkeypatch.setattr(pipeline_helpers, "PROJECT_ROOT", tmp_path)

        async def fake_many(self, blocks, **kw):
            (tmp_path / "arch" / "uarch_specs" / "alpha.md").write_text("I could not write it.")
            return "sorry"

        monkeypatch.setattr(usg.UarchSpecGenerator, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(usg.UarchSpecGenerator, "generate_many", fake_many)
        out = await pipeline_helpers.generate_uarch_specs_single_context([{"name": "alpha"}])
        assert out["missing"] == ["alpha"]
        assert not (tmp_path / "arch" / "uarch_specs" / "alpha.md").exists()
        assert list((tmp_path / "arch" / "uarch_specs").glob("alpha.md.rejected-*"))


class TestGoldenRequiredSignoff:
    def _bd(self, tmp_path, blocks):
        (tmp_path / ".coresmith").mkdir(exist_ok=True)
        (tmp_path / ".coresmith" / "block_diagram.json").write_text(json.dumps({"blocks": blocks}))

    def test_missing_golden_without_exemption_is_listed(self, tmp_path):
        from orchestrator.langgraph import final_report as fr
        self._bd(tmp_path, [
            {"name": "core", "python_source": "inputs/g.py:f"},
            {"name": "fifo", "python_source": ""},
            {"name": "rom", "python_source": "", "golden_exempt": True,
             "no_golden_reason": "contract fixes the ROM contents and latency"},
            {"name": "bad", "golden_exempt": True},  # exemption without a reason does not count
        ])
        rows = [{"name": n} for n in ("core", "fifo", "rom", "bad")]
        assert fr._blocks_without_golden(str(tmp_path), rows) == ["fifo", "bad"]

    def test_no_diagram_means_no_verdict(self, tmp_path):
        from orchestrator.langgraph import final_report as fr
        assert fr._blocks_without_golden(str(tmp_path), [{"name": "x"}]) == []


class TestPrdAnswersFromFeedback:
    def test_feedback_json_counts_as_answers(self):
        # WP-14: the chip lead used to be told to answer in `feedback`; the PRD
        # node only read `answers`, so every answer was dropped (F-2 looped 8x).
        from orchestrator.langgraph import architecture_graph as ag
        src = open(ag.__file__).read()
        assert 'human_response.get("feedback")' in src and "_parsed" in src


class TestValidationAdmissionRegex:
    def test_multiline_assert_with_golden_operand_is_admitted(self):
        import re
        pat = (r"assert\b(?:[^\n]|\n(?=[ \t]))*?(golden|reference|expected_(?:output|bytes|"
               r"bits|value|result|word|state|crc))")
        tb = "async def test_x(dut):\n    assert (\n        out_bytes\n        == golden_bytes\n    ), 'mismatch'\n"
        assert re.search(pat, tb, re.IGNORECASE)
        assert re.search(pat, "    assert out == expected_result\n", re.IGNORECASE)
        assert not re.search(pat, "    assert busy == 0\n\ngolden = 1\n", re.IGNORECASE)
