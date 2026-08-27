"""Tests for the chip-lead loop-breaker batch (arm S/M/U forensic fixes).

Covers: content-free-revise downgrade guard, regeneration-proof revise
constraint pinning, unsupported-action corrective retry, stale contract-audit
archival, and the handshake-role guard in the integration compat checker.
"""

import json
from pathlib import Path

import pytest

from orchestrator.langgraph import integration_helpers, pipeline_graph
from orchestrator.langgraph.integration_helpers import (
    VerilogModule,
    VerilogPort,
    _port_role,
    check_integration_compatibility,
)


class TestContentFreeRevise:
    def test_bare_revise_is_content_free(self):
        assert pipeline_graph._is_content_free_revise({"action": "revise"})

    @pytest.mark.parametrize("field,value", [
        ("feedback", "ports mismatch on cfg bus"),
        ("reasoning", "stale RTL for 3 blocks"),
        ("block_actions", {"blk": "retry"}),
    ])
    def test_substantive_revise_is_not(self, field, value):
        assert not pipeline_graph._is_content_free_revise(
            {"action": "revise", field: value})

    def test_empty_values_still_content_free(self):
        assert pipeline_graph._is_content_free_revise(
            {"action": "revise", "feedback": "", "block_actions": {}})


class TestReviseUarchPinsConstraints:
    """Arm-U critical finding: spec-appended fixes are destroyed by per-tier
    re-spec; the revise must ALSO land in constraints.json."""

    def test_revise_writes_regeneration_proof_constraint(self, tmp_path):
        spec_dir = tmp_path / "arch" / "uarch_specs"
        spec_dir.mkdir(parents=True)
        (spec_dir / "blk_a.md").write_text("# spec\n")
        applied = pipeline_graph._apply_revise_uarch(
            str(tmp_path),
            {"action": "revise", "feedback": "mb_last is a level flag"},
            {"affected_blocks": ["blk_a"]},
            "integration_dv",
        )
        assert applied == ["blk_a"]
        cpath = tmp_path / ".coresmith" / "blocks" / "blk_a" / "constraints.json"
        entries = json.loads(cpath.read_text())
        assert entries[-1]["source"] == "chip_dv_revise"
        assert "mb_last is a level flag" in entries[-1]["rule"]
        assert "INTEGRATION_DV REVISION" in entries[-1]["rule"]

    def test_revise_appends_to_existing_constraints(self, tmp_path):
        spec_dir = tmp_path / "arch" / "uarch_specs"
        spec_dir.mkdir(parents=True)
        (spec_dir / "blk_a.md").write_text("# spec\n")
        bdir = tmp_path / ".coresmith" / "blocks" / "blk_a"
        bdir.mkdir(parents=True)
        (bdir / "constraints.json").write_text(
            json.dumps([{"rule": "keep me", "source": "x"}]))
        pipeline_graph._apply_revise_uarch(
            str(tmp_path),
            {"action": "revise", "feedback": "split the fused bus"},
            {"affected_blocks": ["blk_a"]},
            "validation_dv",
        )
        entries = json.loads((bdir / "constraints.json").read_text())
        assert entries[0]["rule"] == "keep me"
        assert entries[1]["source"] == "chip_dv_revise"


class TestChipLeadUnsupportedActionRetry:
    """Arm-S retro: one unsupported action stranded the run until a human
    daemon restart. The agent now gets exactly one corrective round."""

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_graph, "_CHIP_LEAD_TRIPPED", False)
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CORESMITH_ENABLE_CHIP_LEAD", "1")

    @pytest.mark.asyncio
    async def test_corrective_retry_recovers(self, monkeypatch):
        from orchestrator.langchain.agents import chip_lead_agent
        seen_payloads = []

        async def fake_decide(self, *, payload, prior_decisions=None):
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                return {"action": "revise", "reasoning": "wrong vocab"}
            return {"action": "retry", "reasoning": "corrected"}

        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "__init__",
                            lambda self, model=None: None)
        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "decide", fake_decide)
        monkeypatch.setattr(
            pipeline_graph, "interrupt",
            lambda p: (_ for _ in ()).throw(AssertionError("parked")))

        out = await pipeline_graph._resolve_interrupt(
            {"type": "block_failure", "supported_actions": ["retry", "abort"]})
        assert out["action"] == "retry"
        assert pipeline_graph._CHIP_LEAD_TRIPPED is False
        assert len(seen_payloads) == 2
        assert "action_correction" in seen_payloads[1]
        assert "'revise'" in seen_payloads[1]["action_correction"]

    @pytest.mark.asyncio
    async def test_double_unsupported_trips(self, monkeypatch):
        from orchestrator.langchain.agents import chip_lead_agent

        async def fake_decide(self, *, payload, prior_decisions=None):
            return {"action": "revise", "reasoning": "stubborn"}

        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "__init__",
                            lambda self, model=None: None)
        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "decide", fake_decide)
        parked = []
        monkeypatch.setattr(
            pipeline_graph, "interrupt",
            lambda p: parked.append(p) or {"action": "abort"})

        out = await pipeline_graph._resolve_interrupt(
            {"type": "block_failure", "supported_actions": ["retry", "abort"]})
        assert out == {"action": "abort"}
        assert len(parked) == 1
        assert pipeline_graph._CHIP_LEAD_TRIPPED is True


class TestContractAuditStaleArchive:
    """Arm-M biggest single finding: a timed-out auditor left the previous
    attempt's audit in place and it was read back as the current verdict.
    The live audit path must be cleared (and archived) before the auditor
    runs."""

    @pytest.mark.asyncio
    async def test_prior_audit_archived_before_run(self, tmp_path, monkeypatch):
        audit_dir = tmp_path / ".coresmith" / "contract_audit"
        audit_dir.mkdir(parents=True)
        live = audit_dir / "integration_dv_contract_audit.json"
        live.write_text(json.dumps({"category": "STALE_VERDICT"}))

        from orchestrator.langchain.agents import contract_audit_agent
        state = {}

        class FakeAgent:
            def __init__(self, *a, **k):
                pass

            async def analyze(self, *, stage, project_root, context_path,
                              output_path, callbacks=None):
                state["live_existed_at_call"] = Path(output_path).exists()
                result = {"category": "FRESH", "recommended_action": "retry",
                          "confidence": 0.9}
                Path(output_path).write_text(json.dumps(result))
                return result

        monkeypatch.setattr(contract_audit_agent, "ContractAuditAgent", FakeAgent)

        result = await pipeline_graph._run_top_level_contract_audit(
            stage="integration_dv",
            project_root=str(tmp_path),
            design_name="d",
            top_rtl_path="rtl/top.v",
            testbench_path="tb/tb.py",
            test_count=5,
        )
        assert state["live_existed_at_call"] is False
        assert result["category"] == "FRESH"
        archives = list(audit_dir.glob("integration_dv_contract_audit_*.json"))
        assert len(archives) == 1
        assert json.loads(archives[0].read_text())["category"] == "STALE_VERDICT"


class TestPortRoleGuard:
    """Arm S/M retro: substring resolution paired *_tready with *_tdata and
    the width check reported phantom errors on every integration accept."""

    @pytest.mark.parametrize("name,role", [
        ("s_axis_input_tready", "ready"),
        ("s_axis_sample_tdata", "data"),
        ("m_frame_event_srdy", "valid"),
        ("m_frame_event_drdy", "ready"),
        ("out_last_o", "last"),
        ("busy", None),
        ("clk", None),
    ])
    def test_port_role(self, name, role):
        assert _port_role(name) == role

    def _modules(self):
        src = VerilogModule(name="frame_ingest_store", ports=[
            VerilogPort("s_axis_input_tready", "output", 1),
            VerilogPort("m_axis_sample_tdata", "output", 32),
            VerilogPort("m_axis_sample_tvalid", "output", 1),
        ])
        dst = VerilogModule(name="intra_encode_core", ports=[
            VerilogPort("s_axis_sample_tdata", "input", 32),
            VerilogPort("s_axis_sample_tvalid", "input", 1),
            VerilogPort("s_axis_sample_tready", "output", 1),
        ])
        return {"frame_ingest_store": src, "intra_encode_core": dst}

    def test_cross_role_fuzzy_pair_is_info_not_error(self):
        # No explicit ports: interface name forces substring resolution.
        # Source resolves to the tready (only "input"-term match with
        # prefer_direction output); the destination must not be judged
        # against it as a width error.
        conns = [{
            "from_block": "frame_ingest_store",
            "to_block": "intra_encode_core",
            "from_port": "s_axis_input_tready",
            "to_port": "",
            "interface": "sample_tdata",
            "data_width": 32,
        }]
        mismatches = check_integration_compatibility(conns, self._modules())
        errors = [m for m in mismatches if m.severity == "error"]
        assert errors == []
        assert any(m.issue_type == "role_mismatch_unverifiable"
                   for m in mismatches)

    def test_like_for_like_width_mismatch_still_error(self):
        mods = self._modules()
        mods["intra_encode_core"].ports[0].width = 16  # tdata narrowed
        conns = [{
            "from_block": "frame_ingest_store",
            "to_block": "intra_encode_core",
            "from_port": "m_axis_sample_tdata",
            "to_port": "s_axis_sample_tdata",
            "interface": "sample_stream",
            "data_width": 32,
        }]
        mismatches = check_integration_compatibility(conns, mods)
        assert any(m.issue_type == "width_mismatch" and m.severity == "error"
                   for m in mismatches)

    def test_substring_matcher_respects_roles(self):
        # A lookup whose name implies "data" must not resolve to a ready.
        mod = self._modules()["intra_encode_core"]
        got = integration_helpers._find_port_fuzzy(
            mod, "sample_tdata_bus", "", prefer_direction="input")
        assert got is not None and got.name == "s_axis_sample_tdata"


class TestResolveProbeTop:
    """Arm-U false negative: the synthesizability gate probed the LAST module
    declared in the integration file (a glue adapter), counted 1 cell, and
    failed a chip whose real top synthesizes to 158k cells."""

    REAL = (
        "module h264_encoder_core_top (input clk, output [7:0] out);\n"
        "  syntax_adapter u_adapt(.clk(clk));\n"
        "endmodule\n"
        "module syntax_adapter (input clk);\n"
        "endmodule\n"
    )

    def test_design_name_match_wins_over_last_declared(self):
        assert pipeline_graph._resolve_probe_top(
            "h264_encoder_core_top", self.REAL) == "h264_encoder_core_top"

    def test_uninstantiated_module_wins_when_design_name_absent(self):
        assert pipeline_graph._resolve_probe_top(
            "some_other_design", self.REAL) == "h264_encoder_core_top"

    def test_caravel_wrapper_preferred(self):
        txt = ("module user_project_wrapper (input clk);\nendmodule\n"
               + self.REAL)
        assert pipeline_graph._resolve_probe_top(
            "h264_encoder_core_top", txt) == "user_project_wrapper"

    def test_last_declared_fallback_when_ambiguous(self):
        txt = ("module top_a (input clk);\nendmodule\n"
               "module top_b (input clk);\nendmodule\n")
        assert pipeline_graph._resolve_probe_top("neither", txt) == "top_b"

    def test_single_module_file(self):
        txt = "module only_top (input clk);\nendmodule\n"
        assert pipeline_graph._resolve_probe_top("neither", txt) == "only_top"

    def test_empty_file_falls_back_to_design_name(self):
        assert pipeline_graph._resolve_probe_top("dsn", "") == "dsn"


class TestBoundedValidationMakefile:
    """Arm-U validation loop: the chip lead's bounded-trace/sharded Makefile
    (hot-patched live on e6) is ported into the engine, with the shell-fatal
    COCOTB_TEST_FILTER exclusion regex replaced by a COCOTB_TESTCASE
    include-list that survives every make/shell expansion layer."""

    TB = (
        "@cocotb.test()\n"
        "async def test_boot(dut):\n    pass\n"
        "@cocotb.test()\n"
        "async def test_stream(dut):\n    pass\n"
        "async def _helper(dut):\n    pass\n"
        "@cocotb.test()\n"
        "async def test_wavekit_bounded_semantic_trace(dut):\n    pass\n"
        "@cocotb.test()\n"
        "async def test_frozen_acceptance_matrix(dut):\n"
        "    shard = os.environ.get('ACCEPTANCE_SHARD')\n"
    )

    def _mk(self, scope="validation", tb=None):
        from orchestrator.langgraph.integration_helpers import (
            _compose_dv_makefile)
        return _compose_dv_makefile(
            scope, self.TB if tb is None else tb, "a.v b.v", "chip_top",
            "test_x_validation")

    def test_mission_pass_uses_testcase_list_not_regex(self):
        mk = self._mk()
        assert "COCOTB_TESTCASE = test_boot,test_stream" in mk
        assert "(?!" not in mk  # the shell-fatal lookahead is gone

    def test_no_shell_metachars_in_any_make_assignment(self):
        import re
        mk = self._mk()
        for m in re.finditer(r"^(COCOTB_\w+) = (.*)$", mk, re.M):
            assert not set(m.group(2)) & set("(|)!?'\""), m.group(0)

    def test_bounded_and_sharded_targets_present(self):
        mk = self._mk()
        assert "bounded_waveform:" in mk
        assert "--trace-depth 1" in mk
        assert "acceptance_shard_%" in mk and "$(MAKE) -j4" in mk

    def test_recipe_continuations_intact(self):
        mk = self._mk()
        line = next(l for l in mk.splitlines()
                    if "COCOTB_RESULTS_FILE=trace_results" in l)
        assert line.rstrip().endswith("\\")

    def test_unsharded_tb_gets_bounded_but_no_shards(self):
        tb = ("@cocotb.test()\nasync def test_a(dut):\n    pass\n"
              "@cocotb.test()\n"
              "async def test_wavekit_bounded_semantic_trace(dut):\n    pass\n")
        mk = self._mk(tb=tb)
        assert "bounded_waveform:" in mk
        assert "acceptance_shard" not in mk

    def test_integration_scope_keeps_legacy_makefile(self):
        mk = self._mk(scope="integration")
        assert "WAVES = 1" in mk
        assert "trace-depth" not in mk and "bounded_waveform" not in mk

    def test_plain_validation_tb_keeps_legacy_makefile(self):
        mk = self._mk(tb="@cocotb.test()\nasync def test_a(dut):\n    pass\n")
        assert "WAVES = 1" in mk and "bounded_waveform" not in mk

    def test_mission_line_empty_when_no_specials(self):
        from orchestrator.langgraph.integration_helpers import (
            _mission_testcase_line)
        assert _mission_testcase_line(
            "async def test_a(dut):\n    pass\n") == ""


class TestHarnessAuditFastpath:
    """CORESMITH_HARNESS_AUDIT_FASTPATH both branches: an unambiguous
    harness-class sim failure skips the LLM contract audit (default ON);
    =0 restores the unconditional audit; design-class logs always audit."""

    HARNESS_LOG = (
        "make: Entering directory 'sim_build/validation'\n"
        "bash: -c: line 1: syntax error near unexpected token `('\n"
        "make: *** [sim] Error 2\n"
    )
    DESIGN_LOG = (
        "** test_stream FAIL\n"
        "AssertionError: output beat 69: RTL=0xc1, model=0xc0\n"
    )

    def _install_sentinel_agent(self, monkeypatch, calls):
        from orchestrator.langchain.agents import contract_audit_agent

        real_default = contract_audit_agent.ContractAuditAgent._default_result

        class SentinelAgent:
            _default_result = staticmethod(real_default)

            def __init__(self, *a, **k):
                calls.append("init")

            async def analyze(self, **kw):
                calls.append("analyze")
                result = real_default(kw["stage"], kw["context_path"])
                result["category"] = "LLM_AUDIT_RAN"
                Path(kw["output_path"]).write_text(json.dumps(result))
                return result

        monkeypatch.setattr(
            contract_audit_agent, "ContractAuditAgent", SentinelAgent)

    @pytest.mark.asyncio
    async def test_harness_failure_skips_llm_audit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_HARNESS_AUDIT_FASTPATH", raising=False)
        calls = []
        self._install_sentinel_agent(monkeypatch, calls)
        result = await pipeline_graph._run_top_level_contract_audit(
            stage="validation_dv", project_root=str(tmp_path),
            design_name="d", top_rtl_path="rtl/top.v",
            testbench_path="tb/tb.py", test_count=6,
            sim_log=self.HARNESS_LOG,
        )
        assert calls == []  # no LLM agent constructed
        assert result["category"] == "DV_PROCESS_ERROR"
        assert result["recommended_action"] == "fix_tb"
        assert result["local_fix_possible"] is True
        audit = json.loads(Path(result["audit_path"]).read_text())
        assert audit["category"] == "DV_PROCESS_ERROR"

    @pytest.mark.asyncio
    async def test_env_zero_restores_llm_audit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_HARNESS_AUDIT_FASTPATH", "0")
        calls = []
        self._install_sentinel_agent(monkeypatch, calls)
        result = await pipeline_graph._run_top_level_contract_audit(
            stage="validation_dv", project_root=str(tmp_path),
            design_name="d", top_rtl_path="rtl/top.v",
            testbench_path="tb/tb.py", test_count=6,
            sim_log=self.HARNESS_LOG,
        )
        assert "analyze" in calls
        assert result["category"] == "LLM_AUDIT_RAN"

    @pytest.mark.asyncio
    async def test_design_failure_always_audits(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_HARNESS_AUDIT_FASTPATH", raising=False)
        calls = []
        self._install_sentinel_agent(monkeypatch, calls)
        result = await pipeline_graph._run_top_level_contract_audit(
            stage="integration_dv", project_root=str(tmp_path),
            design_name="d", top_rtl_path="rtl/top.v",
            testbench_path="tb/tb.py", test_count=6,
            sim_log=self.DESIGN_LOG,
        )
        assert "analyze" in calls
        assert result["category"] == "LLM_AUDIT_RAN"

    @pytest.mark.parametrize("needle", [
        "syntax error near unexpected token",
        "No module named test_x",
        "ModuleNotFoundError",
        "verilator: command not found",
    ])
    def test_fingerprints_match(self, needle):
        assert pipeline_graph._harness_failure_fingerprint(
            f"noise\n{needle}\nnoise") is not None

    def test_clean_and_design_logs_do_not_match(self):
        assert pipeline_graph._harness_failure_fingerprint(self.DESIGN_LOG) is None
        assert pipeline_graph._harness_failure_fingerprint("") is None


class TestChipLeadFailureRetry:
    """Arm-F finding: one provider hard-timeout must not trip the sticky
    fail-safe; a single fresh retry absorbs it, two consecutive failures
    still trip."""

    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pipeline_graph, "_CHIP_LEAD_TRIPPED", False)
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CORESMITH_ENABLE_CHIP_LEAD", "1")

    @pytest.mark.asyncio
    async def test_one_failure_retries_and_recovers(self, monkeypatch):
        from orchestrator.langchain.agents import chip_lead_agent
        calls = []

        async def fake_decide(self, *, payload, prior_decisions=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("codex CLI timed out after 900s")
            return {"action": "retry", "reasoning": "recovered"}

        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "__init__",
                            lambda self, model=None: None)
        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "decide",
                            fake_decide)
        monkeypatch.setattr(
            pipeline_graph, "interrupt",
            lambda p: (_ for _ in ()).throw(AssertionError("parked")))
        out = await pipeline_graph._resolve_interrupt(
            {"type": "x", "supported_actions": ["retry", "abort"]})
        assert out["action"] == "retry"
        assert len(calls) == 2
        assert pipeline_graph._CHIP_LEAD_TRIPPED is False

    @pytest.mark.asyncio
    async def test_two_failures_trip(self, monkeypatch):
        from orchestrator.langchain.agents import chip_lead_agent

        async def fake_decide(self, *, payload, prior_decisions=None):
            raise RuntimeError("codex CLI timed out after 900s")

        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "__init__",
                            lambda self, model=None: None)
        monkeypatch.setattr(chip_lead_agent.ChipLeadAgent, "decide",
                            fake_decide)
        parked = []
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda p: parked.append(p) or {"action": "abort"})
        out = await pipeline_graph._resolve_interrupt(
            {"type": "x", "supported_actions": ["retry", "abort"]})
        assert out == {"action": "abort"}
        assert len(parked) == 1
        assert pipeline_graph._CHIP_LEAD_TRIPPED is True
