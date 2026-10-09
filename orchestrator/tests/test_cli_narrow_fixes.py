# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Three generic CLI defects observed in the measured runs: engine progress
lines contaminating ``verify ... --json`` stdout, a missing required tool
input reported as an infrastructure failure, and the ``schema contracts``
example advertising ``flow_control_policy`` as a string where the validator
requires an object."""
from __future__ import annotations

import json
import re
import types

from orchestrator.harness import cli_tool, cli_verify
from orchestrator.harness.tools import schema, validate
from orchestrator.harness.verify import VerifyResult


def test_verify_json_stdout_is_one_document_even_when_the_engine_prints(tmp_path, monkeypatch, capsys):
    from orchestrator.harness import verify as V
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".coresmith").mkdir()

    def chatty_verify_rtl(root, spec, **kw):
        print("[SIM] timeout=600s (default; env=600s declared=0s prior_timeouts=0 cap=3600s) block=tiny", flush=True)
        return VerifyResult(True, verdict="12/12 passed", log_path="/x")
    monkeypatch.setattr(V, "verify_rtl", chatty_verify_rtl)
    args = types.SimpleNamespace(project_root=str(tmp_path), block="tiny", json=True, seed=None, tb=None,
                                 no_equiv=False, lint_only=False, coverage=False)
    rc = cli_verify.cmd_verify_rtl(args)
    out, err = capsys.readouterr()
    assert rc == 0
    doc = json.loads(out)                       # the whole of stdout parses
    assert doc["passed"] is True and doc["verdict"] == "12/12 passed"
    assert "[SIM] timeout" in err               # the progress line went to stderr


def test_missing_required_tool_input_is_a_usage_error_not_infrastructure(tmp_path, monkeypatch, capsys):
    from orchestrator.pdk.base import CheckResult, ToolResult
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    (tmp_path / ".coresmith").mkdir()

    class _Tool:
        def run(self, req):
            script = req.input("script")
            if script is None:
                return ToolResult.from_checks(tool_ok=False, verb="run_sta", design=req.design,
                                              checks=[CheckResult("inputs", "fail", details="run_sta needs --script (STA tcl)")])
            return ToolResult.from_checks(tool_ok=True, verb="run_sta", design=req.design, checks=[CheckResult("sta", "pass")])

    class _Dep:
        name = "fake"

        def supports(self, verb):
            return verb == "run_sta"

        def capabilities(self):
            return {"run_sta"}

        def tool(self, verb):
            return _Tool()
    monkeypatch.setattr(cli_tool, "_get_deployment", lambda: _Dep())
    args = types.SimpleNamespace(verb="run_sta", project_root=str(tmp_path), json=True, design="tiny", rtl=None,
                                 script=None, netlist=None, sdc=None, gds=None, spice=None, liberty=None,
                                 out_dir=None, timeout_s=None)
    rc = cli_tool.cmd_tool_run(args)
    doc = json.loads(capsys.readouterr().out)
    assert rc == cli_tool.EXIT_USAGE == 2
    assert doc["error"] == "TOOL_INPUT_MISSING" and "--script" in doc["required"]
    assert doc["tool_ok"] is False and doc["ok"] is False
    # the audit record says what was missing, not "infrastructure"
    rec = json.loads((tmp_path / ".coresmith" / "tool_runs" / "tool_runs.jsonl").read_text().splitlines()[-1])
    assert rec["metrics"]["error"] == "TOOL_INPUT_MISSING"
    # with the input supplied the same verb runs
    args.script = str(tmp_path / "sta.tcl")
    (tmp_path / "sta.tcl").write_text("# tcl\n")
    assert cli_tool.cmd_tool_run(args) == 0


def test_contracts_schema_example_matches_the_validator():
    text = schema.SCHEMAS["contracts"]
    m = re.search(r'"flow_control_policy":\s*(\S)', text)
    assert m and m.group(1) == "{", "the advertised flow_control_policy must be an object, as the validator requires"
    assert '"flow_control_policy": "..."' not in text
    # the advertised shape passes the validator's type rule; the old string did not
    edge = {"edge_id": "a__p__to__b__q", "producer_block": "a", "producer_port": "p", "consumer_block": "b",
            "consumer_port": "q", "handshake_protocol": "axi_stream", "data_width_bits": 8,
            "flow_control_policy": {"semantics": "skid", "min_buffer_depth_beats": 2}}
    fn = next(getattr(validate, n) for n in dir(validate) if n.startswith("validate_contract"))
    codes = {p.get("code") for p in fn({"contracts": [edge]}, {"blocks": [{"name": "a"}, {"name": "b"}]})}
    assert "CT_POLICY_TYPE" not in codes
    bad = dict(edge, flow_control_policy="...")
    codes = {p.get("code") for p in fn({"contracts": [bad]}, {"blocks": [{"name": "a"}, {"name": "b"}]})}
    assert "CT_POLICY_TYPE" in codes
