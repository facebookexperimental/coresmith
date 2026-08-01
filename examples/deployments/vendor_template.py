# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bring-your-own-PDK/EDA deployment template (commercial-flow skeleton).

Copy this file anywhere, fill in the paths and parsers for your PDK and
tools, then point coresmith at it:

    export CORESMITH_DEPLOYMENT=/abs/path/to/my_deployment.py
    coresmith pdk info
    coresmith tool list

The whole environment lives in this ONE file: the PDK description, the
tool aliases (which binary implements each verb), and the checkers
(Python classes that parse each tool's reports into pass/fail verdicts).
The engine and the agent prompts only ever speak the verb vocabulary
(``run_synth``, ``run_pnr``, ...), so nothing else needs to change.

This skeleton aliases ``run_synth`` -> Synopsys dc_shell and ``run_pnr``
-> Cadence Innovus purely as an illustration; no vendor code or scripts
are included. Every stub raises or returns ``not_run`` until you fill it
in — the base class defaults are honest empties, so anything you omit
degrades to skip/not_run, never to a false green.

Verbs a deployment may implement (see orchestrator/pdk/base.py VERBS):
run_synth, run_pnr, run_drc, run_lvs, run_sta, run_lint, run_sim,
run_gate_sim, gen_macro. Omit a verb and `coresmith tool <verb>` exits 4
(honest capability skip) and downstream gates record not_run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

from orchestrator.pdk.base import (
    Checker,
    CheckResult,
    Deployment,
    EdaTool,
    ToolRequest,
    ToolResult,
)
from orchestrator.pdk.pdk_config import CornerConfig, PDKConfig

# ---------------------------------------------------------------------------
# 1. The PDK: paths, corners, cell families.
#    Use {pdk_root} placeholders; PDK_ROOT env resolves them at load.
# ---------------------------------------------------------------------------
_PDK = PDKConfig(
    name="my_pdk",                       # e.g. "asap7"
    process_nm=7,
    std_cell_library="my_stdcell_lib",   # e.g. "asap7sc7p5t"
    site_name="my_site",
    supply_voltage=0.7,
    default_corner="tt_25C_0v70",
    corners={
        "tt_25C_0v70": CornerConfig(
            name="tt_25C_0v70",
            liberty="{pdk_root}/lib/my_stdcell_tt.lib",
            temperature=25,
            voltage=0.7,
        ),
    },
    lef_path="{pdk_root}/lef/my_stdcell.lef",
    tech_lef_path="{pdk_root}/lef/my_tech.tlef",
    # Optional: cells=CellConfig(...), pnr=PnrConfig(...) — see
    # orchestrator/pdk/configs/sky130.yaml for the full shape. Tools that
    # generate their own scripts (like the sky130 OpenROAD flow) read them;
    # a flow whose scripts you author externally can leave them unset.
)


# ---------------------------------------------------------------------------
# 2. Checkers: one class per report your tools emit.
#    Three-state contract: pass / fail / not_run (missing report). A
#    blocking not_run FAILS the verb — never let a lost report look green.
# ---------------------------------------------------------------------------
class DcQorChecker(Checker):
    """Parse dc_shell's qor/area reports into cells/wns metrics."""

    name = "synth_qor"
    blocking = True

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        rpt = run_dir / f"{req.design}.qor.rpt"
        if not rpt.exists():
            return CheckResult(self.name, "not_run",
                               details=f"missing {rpt.name}")
        # TODO: parse cell count / WNS out of your QoR report format, e.g.:
        # wns_ns = _parse_wns(rpt.read_text())
        # status = "pass" if wns_ns >= 0 else "fail"
        return CheckResult(self.name, "fail",
                           details="TODO: implement QoR parsing")


class InnovusDrcChecker(Checker):
    """Count post-route DRC violations from Innovus' verify_drc report."""

    name = "route_drc"
    blocking = True

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        rpt = run_dir / f"{req.design}.drc.rpt"
        if not rpt.exists():
            return CheckResult(self.name, "not_run",
                               details=f"missing {rpt.name}")
        # TODO: parse violation count; 0 -> pass, >0 -> fail with count in
        # metrics={"violations": n} so gates/telemetry can read it.
        return CheckResult(self.name, "fail",
                           details="TODO: implement DRC report parsing")


# ---------------------------------------------------------------------------
# 3. Tools: one class per verb, aliasing the verb to your binary.
#    The agent authors/repairs the script; run() executes it and runs the
#    checkers. Infra problems (binary missing, timeout) -> tool_ok=False
#    (CLI exit 3); the tool ran but a checker failed -> exit 1.
# ---------------------------------------------------------------------------
class DcSynth(EdaTool):
    verb: ClassVar[str] = "run_synth"

    def run(self, req: ToolRequest) -> ToolResult:
        out_dir = req.out_dir or Path.cwd()
        script = req.input("script")
        if script is None:
            return ToolResult.from_checks(
                tool_ok=False, verb=self.verb, design=req.design,
                checks=[CheckResult("invocation", "not_run",
                                    details="run_synth requires --script")],
            )
        cmd = ["dc_shell", "-batch", "-f", str(script)]  # TODO: your binary
        log = out_dir / "synth.log"
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                                  timeout=req.timeout_s or 3600,
                                  cwd=out_dir)
            log.write_text(proc.stdout + proc.stderr)
            tool_ok = proc.returncode == 0
        except FileNotFoundError:
            return ToolResult.from_checks(
                tool_ok=False, verb=self.verb, design=req.design,
                checks=[CheckResult("invocation", "not_run",
                                    details="dc_shell not on PATH")],
            )
        except subprocess.TimeoutExpired:
            return ToolResult.from_checks(
                tool_ok=False, verb=self.verb, design=req.design,
                checks=[CheckResult("invocation", "not_run",
                                    details="timeout")],
            )
        checks = [c.check(req, out_dir) for c in self.checkers()]
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks, log_path=log,
            verb=self.verb, design=req.design,
            artifacts={"netlist": out_dir / f"{req.design}_netlist.v"},
        )

    def checkers(self) -> list[Checker]:
        return [DcQorChecker()]

    def prompt_notes(self) -> str:
        # Shown to the script-authoring agent as {tool_notes}. Put your
        # flow's recipe here, NOT in the engine prompt files.
        return ("run_synth wraps dc_shell -batch -f <script>. Author a dc "
                "TCL that reads the RTL, links, compiles, and writes "
                "<design>_netlist.v plus <design>.qor.rpt into --out-dir.")


class InnovusPnr(EdaTool):
    verb: ClassVar[str] = "run_pnr"

    def run(self, req: ToolRequest) -> ToolResult:
        # Same shape as DcSynth.run: invoke `innovus -batch -files <script>`,
        # then run the checkers over --out-dir.
        return ToolResult.from_checks(
            tool_ok=False, verb=self.verb, design=req.design,
            checks=[CheckResult("invocation", "not_run",
                                details="TODO: implement Innovus invocation")],
        )

    def checkers(self) -> list[Checker]:
        return [InnovusDrcChecker()]

    def prompt_notes(self) -> str:
        return ("run_pnr wraps Innovus in batch mode. Author a floorplan-"
                "to-route TCL writing <design>.def, <design>.drc.rpt, and "
                "timing reports into --out-dir.")


# ---------------------------------------------------------------------------
# 4. The deployment: bind PDK + verb->tool map.
#    Optional overrides for full engine parity (gate-sim cell models, macro
#    discovery, GDS layer map): cell_model_files / udp_shim_source /
#    sim_defines / macro_search_paths / macro_defaults / gds_layer_map /
#    data_dir — omit what you don't have; omissions degrade honestly.
# ---------------------------------------------------------------------------
class VendorDeployment(Deployment):
    name: ClassVar[str] = "vendor-template"

    def __init__(self) -> None:
        self._pdk = _PDK
        self._tools: dict[str, EdaTool] = {
            "run_synth": DcSynth(self),
            "run_pnr": InnovusPnr(self),
            # Reuse the open-source lint tool unchanged if verilator works
            # for your RTL dialect — deployments may mix vendor and OSS:
            # "run_lint": RunLintVerilator(self),  # from deployments.sky130
        }

    @property
    def pdk(self) -> PDKConfig:
        return self._pdk

    def tools(self) -> dict[str, EdaTool]:
        return self._tools


# The registry loads this module attribute when CORESMITH_DEPLOYMENT points
# at this file.
DEPLOYMENT = VendorDeployment()
