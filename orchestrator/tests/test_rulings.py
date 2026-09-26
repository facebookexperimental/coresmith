# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C2: operator rulings -- an additive policy channel that is not tampering."""
from __future__ import annotations

import argparse
import json
import stat

import pytest

from orchestrator.state_store import rulings as R
from orchestrator.state_store.project_db import open_project


def _proj(tmp_path, req="Target 64 MHz SoC clock; 128 MiB RAM; 320x240 at 30 fps.\n"):
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs" / "requirements.md").write_text(req)
    db = open_project(tmp_path)
    db.begin_run("r1")
    return db


class TestRows:
    def test_scopes_precedence_supersede_revoke(self, tmp_path):
        db = _proj(tmp_path)
        g = db.add_ruling("global", "Prefer flops under 1 KiB")
        a = db.add_ruling("arch", "Use a fabric primitive for the bus")
        b = db.add_ruling("block:core", "Two-stage TLB lookup")
        e = db.add_ruling("edge:core__x__to__l1i__y", "N=1 response")
        assert [r["id"] for r in db.rulings_for()] == [g]
        assert [r["id"] for r in db.rulings_for(arch=True)] == [g, a]
        assert [r["id"] for r in db.rulings_for(block="core", edge_ids=["core__x__to__l1i__y"])] == [g, b, e]
        b2 = db.add_ruling("block:core", "Three-stage TLB lookup", supersedes_id=b)
        ids = [r["id"] for r in db.rulings_for(block="core")]
        assert b not in ids and b2 in ids
        assert db.revoke_ruling(b2, "wrong")
        assert not db.revoke_ruling(b2, "again")
        assert [r["id"] for r in db.rulings_for(block="core")] == [g]
        with pytest.raises(ValueError):
            db.add_ruling("planet", "x")

    def test_block_ruling_is_a_persistent_constraint(self, tmp_path):
        from orchestrator.langgraph import pipeline_graph as pg
        db = _proj(tmp_path)
        rid = db.add_ruling("block:core", "Keep the L1D blocking", rationale="area")
        rows = db.constraints("core")
        assert rows and rows[0]["source"] == "operator_ruling" and rows[0]["ruling_id"] == rid
        assert "operator_ruling" in pg._PERSISTENT_CONSTRAINT_SOURCES
        db.prune_constraints("core", pg._PERSISTENT_CONSTRAINT_SOURCES)
        assert db.constraints("core")            # survives the per-round prune
        db.revoke_ruling(rid, "no")
        assert db.constraints("core") == []      # revoke drops it

    def test_conflict_validator_flags_but_records(self, tmp_path):
        db = _proj(tmp_path)
        rid = db.add_ruling("global", "Run the SoC at 50 MHz")
        r = db.ruling(rid)
        assert r["conflicts"] and r["conflicts"][0]["unit"].lower() == "mhz"
        assert r["conflicts"][0]["requirement_values"] == ["64"]
        assert db.ruling(db.add_ruling("global", "Keep 64 MHz"))["conflicts"] == []


class TestRenderAndLedger:
    def test_section_is_deterministic_and_ledgered(self, tmp_path):
        db = _proj(tmp_path)
        assert R.render_rulings_section(db, consumer="x") == ""   # byte-identical prompts
        db.add_ruling("global", "Prefer flops", rationale="no macros")
        db.add_ruling("block:core", "Blocking L1D")
        s1 = R.render_rulings_section(db, consumer="rtl_generator", block="core", attempt=1)
        s2 = R.render_rulings_section(db, consumer="rtl_generator", block="core", attempt=1)
        assert s1 == s2
        assert "## Operator rulings (binding)" in s1 and "[R1, global] Prefer flops — no macros" in s1
        assert "[R2, block:core] Blocking L1D" in s1
        uses = db.ruling_uses()
        assert len(uses) == 4 and {u["consumer"] for u in uses} == {"rtl_generator"}
        assert R.rulings_section(str(tmp_path), consumer="y", block="other").count("[R") == 1
        assert R.rulings_section("", consumer="y") == "" and R.rulings_section(".", consumer="y") == ""

    def test_view_is_read_only_and_outside_inputs(self, tmp_path):
        from orchestrator.state_store.trust import (
            _oracle_files,
            capture_run_baseline,
            check_oracle_manifest,
        )
        db = _proj(tmp_path)
        capture_run_baseline(tmp_path)
        db.add_ruling("global", "Prefer flops")
        view = db.export_rulings_view()
        assert view == tmp_path / ".coresmith" / "OPERATOR_RULINGS.md"
        assert not (view.stat().st_mode & stat.S_IWUSR)
        assert "Prefer flops" in view.read_text()
        assert all(".coresmith" not in str(p) for p in _oracle_files(tmp_path))
        verdict = check_oracle_manifest(tmp_path)
        assert verdict.get("ok") is True, verdict   # adding a ruling is NOT tampering

    def test_prompts_carry_the_section(self, tmp_path, monkeypatch):
        db = _proj(tmp_path)
        db.add_ruling("block:core", "Blocking L1D")
        from orchestrator.langchain.agents import rtl_generator as rg
        msg = rg.build_user_message("core", description="d", project_root=str(tmp_path))
        assert "Blocking L1D" in msg
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        assert "Blocking L1D" in R.rulings_section_env(consumer="fix_lint", block="core")
        assert "Blocking L1D" not in R.rulings_section_env(consumer="fix_lint", block="other")


class TestResolveInterrupts:
    def test_interrupt_ref_resolves_a_block_park(self, tmp_path):
        db = _proj(tmp_path)
        iid, _ = db.park_interrupt({"type": "human_intervention_needed", "block_name": "core",
                                    "supported_actions": ["retry", "add_constraint", "skip"]},
                                   graph="pipeline", node="ask_human")
        rid = db.add_ruling("block:core", "Split the FPU normalize stage",
                            question_ref=f"interrupt:{iid}")
        assert R.apply_ruling_to_interrupts(db, db.ruling(rid)) == [iid]
        row = db.interrupt(iid)
        assert row["status"] == "resolved" and row["resolved_by"] == f"ruling:{rid}"
        assert row["resolution"]["action"] == "add_constraint"
        assert row["resolution"]["constraint"] == "Split the FPU normalize stage"

    def test_prd_questions_resolve_when_every_id_is_answered(self, tmp_path):
        db = _proj(tmp_path)
        payload = {"type": "prd_questions", "questions": [
            {"id": "coherence", "text": "snoop or directory?"},
            {"id": "gpu_isa", "text": "?"}, {"id": "video", "text": "?"}]}
        iid, _ = db.park_interrupt(payload, graph="architecture", node="Escalate PRD")
        r1 = db.add_ruling("arch", "Snooping MESI", question_ref="prd:coherence")
        assert R.apply_ruling_to_interrupts(db, db.ruling(r1)) == []
        assert db.interrupt(iid)["status"] == "pending"
        r2 = db.add_ruling("arch", "RV32EM-based SIMT", question_ref="prd:gpu_isa")
        assert R.apply_ruling_to_interrupts(db, db.ruling(r2)) == []
        r3 = db.add_ruling("arch", "Designer's choice", question_ref="prd:*")
        assert R.apply_ruling_to_interrupts(db, db.ruling(r3)) == [iid]
        res = db.interrupt(iid)["resolution"]
        assert res["action"] == "continue"
        assert res["answers"] == {"coherence": "Snooping MESI", "gpu_isa": "RV32EM-based SIMT",
                                  "video": "Designer's choice"}

    def test_block_kind_ref(self, tmp_path):
        db = _proj(tmp_path)
        a, _ = db.park_interrupt({"type": "dv_failure", "block_name": "core",
                                  "supported_actions": ["retry", "skip"]}, graph="pipeline", node="n")
        b, _ = db.park_interrupt({"type": "dv_failure", "block_name": "gpu",
                                  "supported_actions": ["retry", "skip"]}, graph="pipeline", node="n")
        rid = db.add_ruling("block:core", "Retry with the fix", question_ref="block:core:dv_failure")
        assert R.apply_ruling_to_interrupts(db, db.ruling(rid)) == [a]
        assert db.interrupt(a)["resolution"]["action"] == "retry"   # first supported action
        assert db.interrupt(b)["status"] == "pending"


class TestSurfaces:
    def test_cli_add_list_revoke(self, tmp_path):
        from orchestrator.harness import cli
        db = _proj(tmp_path)
        parser = argparse.ArgumentParser()
        cli.register_subcommands(parser.add_subparsers())
        args = parser.parse_args(["ruling", "add", "--project-root", str(tmp_path),
                                  "--scope", "global", "--text", "Prefer flops", "--json"])
        assert cli.cmd_ruling(args) == 0
        assert (tmp_path / ".coresmith" / "OPERATOR_RULINGS.md").exists()
        args = parser.parse_args(["ruling", "list", "--project-root", str(tmp_path)])
        assert cli.cmd_ruling(args) == 0
        args = parser.parse_args(["ruling", "revoke", "1", "--project-root", str(tmp_path),
                                  "--reason", "x"])
        assert cli.cmd_ruling(args) == 0
        assert db.rulings() == []
        args = parser.parse_args(["ruling", "add", "--project-root", str(tmp_path),
                                  "--scope", "nope", "--text", "x"])
        assert cli.cmd_ruling(args) != 0

    @pytest.mark.asyncio
    async def test_http_and_mcp(self, tmp_path, monkeypatch):
        from orchestrator import mcp_server as m
        from orchestrator.daemon import server as ds
        monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(m, "_project_root", lambda: str(tmp_path))
        db = _proj(tmp_path)
        iid, _ = db.park_interrupt({"type": "dv_failure", "block_name": "core",
                                    "supported_actions": ["retry", "skip"]}, graph="pipeline", node="n")

        class _Running:
            def done(self):
                return False
        monkeypatch.setattr(ds._pipeline, "task", _Running())
        out = await ds.rulings_add(ds.RulingRequest(scope="block:core", text="Retry it",
                                                    question_ref=f"interrupt:{iid}"))
        assert out["resolved_interrupts"] == [iid] and out["applied_now"] is False
        got = await ds.rulings_list()
        assert got["count"] == 1
        out = json.loads(await m.add_ruling(scope="global", text="Prefer flops"))
        assert out["id"] == 2
        assert json.loads(await m.list_rulings())["count"] == 2
        assert json.loads(await m.revoke_ruling(2, "x"))["revoked"] is True
        await ds.rulings_revoke(1, ds.RevokeRulingRequest(reason="done"))
        assert db.rulings() == []

    def test_final_report_section(self, tmp_path):
        from orchestrator.langgraph import final_report as fr
        db = _proj(tmp_path)
        db.add_ruling("global", "Prefer flops")
        R.render_rulings_section(db, consumer="rtl_generator", block="core")
        rows = fr._operator_rulings(str(tmp_path))
        assert rows[0]["uses"] == 1 and rows[0]["consumers"] == ["rtl_generator"]
        md = fr.render_markdown({"design_name": "chip", "signoff": {"status": "FAIL"},
                                 "operator_rulings": rows})
        assert "## Operator rulings applied" in md and "Prefer flops" in md
