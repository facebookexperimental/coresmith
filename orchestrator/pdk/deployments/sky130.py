# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SkyWater Sky130 HD -- the reference deployment.

This module is the single source of truth for the Sky130 backend PDK paths and
EDA tool-binary resolution. The constants and ``_resolve_tool`` here were
previously hardcoded in ``orchestrator/langgraph/backend_helpers.py`` (lines
44-106) and ``pipeline_helpers`` (the liberty finder); ``backend_helpers`` now
re-exports them from here so every call site keeps working unchanged.

Resolution order for a tool binary (first match wins), preserved from the old
``_resolve_tool``::

    1. ``CORESMITH_BACKEND_<NAME>`` env var (nix shellHook / Docker set these)
    2. ``backend.<config_key>`` in ``orchestrator/config.yaml``
    3. ``scripts/<tool>-nix.sh`` under the project root
    4. the bare default (OS resolves via ``$PATH``)

PDK paths derive from ``PDK_ROOT`` (env ``PDK_ROOT`` or ``<project>/.pdk``) and
the ``sky130A`` / ``sky130B`` variant that exists on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from orchestrator.langgraph.pipeline_helpers import PROJECT_ROOT, load_config
from orchestrator.pdk.base import (
    Checker,
    CheckResult,
    Deployment,
    EdaTool,
    ToolRequest,
    ToolResult,
)
from orchestrator.pdk.checkers import (
    LintChecker,
    LogicDepthChecker,
    LvsMatchChecker,
    MagicDrcChecker,
    PnrReportsChecker,
    RouteDrcChecker,
    StaChecker,
    SynthStatChecker,
)
from orchestrator.pdk.pdk_config import PDKConfig

_STD_CELL = "sky130_fd_sc_hd"
_CONFIG_YAML = Path(__file__).resolve().parents[1] / "configs" / "sky130.yaml"
# Sidecar TCL/data templates live in the CODE tree (next to orchestrator/), not
# a run's project root -- anchor to this file so resolution is independent of
# PROJECT_ROOT (which is a per-run directory during backend runs).
_DATA_DIR = Path(__file__).resolve().parents[2] / "pdk_templates" / "sky130"

# Log substrings that mean "the tool never really ran" (infra, exit 3) rather
# than "the tool ran and reported a problem" (verb fail, exit 1).
_INFRA_MARKERS = (
    "not installed",
    "not found",
    "did not terminate within",
    "timed out",
    "timeout",
)


def _is_infra_failure(log: str) -> bool:
    low = (log or "").lower()
    return any(m in low for m in _INFRA_MARKERS)


# ---------------------------------------------------------------------------
# PDK path resolution (absorbed from backend_helpers.py:44-63 + pipeline
# _find_liberty_file). Parametrized on ``pdk_root`` so tests can resolve against
# a synthetic sky130A/sky130B tree without the real PDK.
# ---------------------------------------------------------------------------
def _current_pdk_root() -> Path:
    """Resolve PDK_ROOT identically to ``pipeline_helpers.PDK_ROOT``."""
    return Path(os.environ.get("PDK_ROOT", "").strip() or (PROJECT_ROOT / ".pdk"))


def _pdk_variant(pdk_root: Path) -> str:
    """Return the PDK variant directory name (sky130A or sky130B)."""
    for v in ("sky130A", "sky130B"):
        if (pdk_root / v).is_dir():
            return v
    return "sky130A"


@dataclass(frozen=True)
class Sky130Paths:
    """The eight PDK path constants, resolved for one ``pdk_root``."""

    pdk_root: Path
    variant: str
    pdk_path: Path
    tech_lef: Path
    cell_lef: Path
    liberty: Path
    cell_gds: Path
    cell_spice: Path
    magic_rc: Path
    netgen_setup: Path
    rcx_rules: Path


def _resolve_paths(pdk_root: Path) -> Sky130Paths:
    """Compute the Sky130 HD path set. Byte-identical to the old backend_helpers
    formula for a given ``pdk_root`` (guarded by the parity test)."""
    var = _pdk_variant(pdk_root)
    p = pdk_root / var
    ref = p / "libs.ref" / _STD_CELL
    tech = p / "libs.tech"
    return Sky130Paths(
        pdk_root=pdk_root,
        variant=var,
        pdk_path=p,
        tech_lef=ref / "techlef" / f"{_STD_CELL}__nom.tlef",
        cell_lef=ref / "lef" / f"{_STD_CELL}.lef",
        liberty=ref / "lib" / f"{_STD_CELL}__tt_025C_1v80.lib",
        cell_gds=ref / "gds" / f"{_STD_CELL}.gds",
        cell_spice=ref / "spice" / f"{_STD_CELL}.spice",
        magic_rc=tech / "magic" / f"{var}.magicrc",
        netgen_setup=tech / "netgen" / "setup.tcl",
        rcx_rules=tech / "rcx" / "sky130hd_rcx_patterns.rules",
    )


def _resolve_tool(config_key: str, default_script: str) -> str:
    """Resolve an EDA tool binary path (backend_helpers._resolve_tool, verbatim).

    1. ``CORESMITH_BACKEND_<NAME>`` env var.
    2. ``backend.<config_key>`` in ``orchestrator/config.yaml``.
    3. ``default_script`` relative to the project root.
    4. ``default_script`` as-is (OS resolves via ``$PATH``).
    """
    env_key = "CORESMITH_BACKEND_" + config_key.removesuffix("_binary").upper()
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val

    try:
        cfg = load_config()
        backend = cfg.get("backend", {})
        path = backend.get(config_key, "")
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return str(p)
    except Exception:  # noqa: BLE001
        pass
    p = PROJECT_ROOT / default_script
    if p.exists():
        return str(p)
    return default_script


def _resolve_yosys() -> str:
    """Resolve the yosys binary for engine-run synthesis.

    Unlike openroad/magic/netgen (backend-only, invoked inside ``nix develop``),
    yosys is executed DIRECTLY by ``synthesize_block`` for PDK-free generic synth
    on hosts without Nix. To preserve today's bare-``yosys`` behavior there while
    still honoring the flake/Docker ``CORESMITH_BACKEND_YOSYS`` override, we
    resolve env -> PATH -> ``scripts/yosys-nix.sh`` fallback. (The old code
    ignored ``_resolve_tool`` entirely and always used bare ``yosys``; routing
    it through the deployment closes that inconsistency without regressing
    non-Nix boxes, whose ``config.yaml`` points yosys at the Nix wrapper.)
    """
    env_val = os.environ.get("CORESMITH_BACKEND_YOSYS", "").strip()
    if env_val:
        return env_val
    which = shutil.which("yosys")
    if which:
        return which
    return _resolve_tool("yosys_binary", "scripts/yosys-nix.sh")


# ---------------------------------------------------------------------------
# TCL table formatters (PR5). These render the PDK-derived data
# (``PDKConfig.cells`` / ``PDKConfig.pnr``) into the exact TCL tokens the
# reference OpenROAD flow uses. Kept as pure functions of a PDKConfig so the
# generated script is byte-identical to the pre-move backend_helpers output
# (guarded by test_pdk_tcl_golden). A different PDK supplies its own cells/pnr
# in its config and these formatters reproduce its flow unchanged.
# ---------------------------------------------------------------------------
def _tracks_tcl(pnr) -> str:
    """The ``make_tracks`` block. Layer is left-justified to 4 (``li1 ``) so the
    two-space alignment before ``-x_offset`` matches the reference verbatim."""
    return "".join(
        f"make_tracks {t['layer']:<4} -x_offset {t['x_offset']} "
        f"-x_pitch {t['x_pitch']} -y_offset {t['y_offset']} "
        f"-y_pitch {t['y_pitch']}\n"
        for t in pnr.tracks
    )


def _fillers_tcl(cells) -> str:
    return " ".join(cells.fillers)


def _clkbuf_list_tcl(cells) -> str:
    return " ".join(cells.clkbuf_list)


def _dont_use_tcl(cells) -> str:
    return " ".join(cells.dont_use)


# ---------------------------------------------------------------------------
# Minimal checkers (PR1). Verdicts are largely composed inline in the tool
# ``run()`` bodies (which hold the helper result dicts); these classes carry the
# fail-closed "report present?" rule and describe the tool for ``tool list``.
# PR3 promotes the existing report parsers into full Checker classes.
# ---------------------------------------------------------------------------
class OutputArtifactChecker(Checker):
    """A named output must exist on disk; missing => ``not_run`` (fail-closed)."""

    def __init__(self, name: str, artifact_key: str) -> None:
        self.name = name
        self.artifact_key = artifact_key

    def check(self, req: ToolRequest, run_dir: Path) -> CheckResult:
        # Best-effort: the concrete path is known in run(); here we only assert
        # the run dir produced *something* for the keyed artifact.
        hits = list(run_dir.glob(f"*{self.artifact_key}*")) if run_dir else []
        if hits:
            return CheckResult(self.name, "pass", metrics={"count": len(hits)})
        return CheckResult(
            self.name, "not_run",
            details=f"no {self.artifact_key} artifact in {run_dir}",
        )


def _run_cmd(cmd: list[str], timeout: int,
             cwd: str | None = None) -> tuple[int, str, str, str]:
    """Run a subprocess; return (rc, stdout, stderr, infra_reason).

    ``infra_reason`` is non-empty only for a genuine infra failure (timeout /
    binary missing) so callers can map it to exit-3 rather than a verb fail.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout, r.stderr, ""
    except subprocess.TimeoutExpired:
        return 124, "", "", f"did not terminate within {timeout}s (timeout)"
    except (FileNotFoundError, OSError) as exc:
        return 127, "", "", f"tool not found: {exc}"


def _synth_out_dir(req: ToolRequest) -> Path:
    return Path(req.out_dir) if req.out_dir else (
        PROJECT_ROOT / "syn" / "output" / (req.design or "block"))


def _artifact_check(name: str, path: Path | None) -> CheckResult:
    """Compose an inline artifact-presence check from a known path."""
    if path is not None and Path(path).exists():
        return CheckResult(name, "pass", metrics={"path": str(path)})
    return CheckResult(name, "not_run", details=f"missing: {path}")


# ---------------------------------------------------------------------------
# Tool classes -- behavior-preserving wrappers over the existing engine helpers.
# All engine imports are LAZY (inside run()) to avoid an import cycle with
# backend_helpers (which re-exports this module's constants at load time).
# ---------------------------------------------------------------------------
class RunSynthYosys(EdaTool):
    verb: ClassVar[str] = "run_synth"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.pipeline_helpers import synthesize_block

        # Agent-authored script path: run the caller's .ys VERBATIM (the flat-top
        # backend synth authors a custom script -- macro directives, lpflow-strip,
        # etc.). No script -> the deployment generates + runs its own recipe via
        # synthesize_block (the per-block frontend path).
        script = req.input("script")
        if script is not None:
            return self._run_script(req, script)

        rtl = req.input("rtl")
        if rtl is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail", details="run_synth needs --rtl or --script")],
                verb=self.verb, design=req.design,
            )
        clock = float(req.params.get("clock_mhz", 50.0))
        res = synthesize_block(
            {"name": req.design}, str(rtl), target_clock_mhz=clock,
            attempt=int(req.params.get("attempt", 1)),
        )
        success = bool(res.get("success"))
        tool_ok = success or not _is_infra_failure(res.get("log", ""))
        netlist = res.get("netlist_path")
        checks = [
            CheckResult("synth_returncode", "pass" if success else "fail",
                        details=res.get("log", "")[-400:]),
            _artifact_check("netlist", Path(netlist) if netlist else None),
        ]
        # Cell / FF / area metrics come from the SynthStatChecker (advisory),
        # which wraps ppa_check.count_cells_from_stat -- the parser that DOES
        # understand the Yosys 0.65 box-format stat ("N cells") that
        # synthesize_block's own inline loop misses. Fall back to the helper's
        # own numbers if no report is on disk to parse.
        report = res.get("report_path")
        stat = SynthStatChecker()
        stat_res = stat.check(
            req, Path(report).parent if report else (req.out_dir or PROJECT_ROOT))
        checks.append(stat_res)
        cells = int(stat_res.metrics.get("cells", 0) or res.get("gate_count", 0) or 0)
        ff = int(stat_res.metrics.get("ff_count", 0) or res.get("ff_count", 0) or 0)
        area = float(stat_res.metrics.get("chip_area_um2", 0.0)
                     or res.get("chip_area_um2", 0.0) or 0.0)
        metrics = {
            "cells": cells,
            "gate_count": cells or res.get("gate_count", 0),
            "ff_count": ff,
            "chip_area_um2": area,
        }
        artifacts = {
            k: Path(res[v]) for k, v in
            (("netlist", "netlist_path"), ("sdc", "sdc_path"), ("report", "report_path"))
            if res.get(v)
        }
        lp = res.get("log_path")
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks, artifacts=artifacts,
            log_path=Path(lp) if lp else None, verb=self.verb,
            design=req.design, extra_metrics=metrics,
        )

    def _run_script(self, req: ToolRequest, script: Path) -> ToolResult:
        """Run a caller-authored yosys script verbatim and parse its stat."""
        out_dir = _synth_out_dir(req)
        out_dir.mkdir(parents=True, exist_ok=True)
        rc, stdout, stderr, infra = _run_cmd(
            [_resolve_yosys(), "-s", str(script)],
            timeout=req.timeout_s or 900, cwd=str(PROJECT_ROOT))
        report = out_dir / f"{req.design or 'synth'}_report.txt"
        report.write_text(stdout + "\n" + stderr)
        success = rc == 0 and not infra
        tool_ok = success or not (infra or _is_infra_failure(stderr))
        checks = [CheckResult("synth_returncode", "pass" if success else "fail",
                              details=(stderr or stdout)[-400:])]
        stat_res = SynthStatChecker().check(req, out_dir)
        checks.append(stat_res)
        nets = sorted(out_dir.glob("*netlist*.v")) or sorted(out_dir.glob("*.v"))
        if nets:
            checks.append(_artifact_check("netlist", nets[0]))
        metrics = {
            "cells": int(stat_res.metrics.get("cells", 0) or 0),
            "gate_count": int(stat_res.metrics.get("cells", 0) or 0),
            "ff_count": int(stat_res.metrics.get("ff_count", 0) or 0),
            "chip_area_um2": float(stat_res.metrics.get("chip_area_um2", 0.0) or 0.0),
        }
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks,
            artifacts={"netlist": nets[0]} if nets else {},
            log_path=report, verb=self.verb, design=req.design,
            extra_metrics=metrics,
        )

    def checkers(self) -> list[Checker]:
        return [SynthStatChecker(), LogicDepthChecker(),
                OutputArtifactChecker("netlist_present", "netlist")]

    def prompt_notes(self) -> str:
        return (
            "Yosys targeting the sky130 high-density standard-cell library.\n"
            "- PDK-mapped flow (liberty present): read_verilog -> "
            "hierarchy -check -top <top> -> proc; opt; fsm; opt; memory; opt -> "
            "techmap; opt -> dfflibmap -liberty <lib>; abc -liberty <lib>; clean; "
            "opt_clean -purge -> write_verilog -noattr; stat -liberty <lib>.\n"
            "- PDK-free generic flow (no liberty on this host): map with "
            "`abc -g AND,OR,XOR,MUX` then a plain `stat`; this still proves the "
            "design elaborates and maps to terminating gates.\n"
            "- Parse the final `stat` for the cell count (the module total is a "
            "`N cells` line on newer Yosys) and area."
        )


class RunLintVerilator(EdaTool):
    verb: ClassVar[str] = "run_lint"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.pipeline_helpers import lint_rtl

        rtl = req.input("rtl")
        if rtl is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail", details="run_lint needs --rtl")],
                verb=self.verb, design=req.design,
            )
        res = lint_rtl(str(rtl), req.design or Path(rtl).stem)
        clean = bool(res.get("clean"))
        msg = res.get("errors", "") or res.get("warnings", "")
        tool_ok = clean or not _is_infra_failure(msg)
        checks = [CheckResult("lint", "pass" if clean else "fail",
                              details=msg[-400:])]
        lp = res.get("log_path")
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks,
            log_path=Path(lp) if lp else None,
            verb=self.verb, design=req.design,
            extra_metrics={"clean": clean},
        )

    def checkers(self) -> list[Checker]:
        return [LintChecker()]

    def prompt_notes(self) -> str:
        return "verilator --lint-only -Wall -Wno-fatal; %Error tokens fail the lint."


class RunPnrOpenroad(EdaTool):
    verb: ClassVar[str] = "run_pnr"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.backend_helpers import run_openroad, run_pnr_flow

        out_dir = str(req.out_dir or (PROJECT_ROOT / "pnr" / "output" / req.design))
        script = req.input("script")
        if script is not None:
            res = run_openroad(str(script), req.design, "pnr",
                               timeout=req.timeout_s or 1800)
            success = bool(res.get("success"))
        else:
            netlist, sdc = req.input("netlist"), req.input("sdc")
            if netlist is None or sdc is None:
                return ToolResult.from_checks(
                    tool_ok=False,
                    checks=[CheckResult("inputs", "fail",
                                        details="run_pnr needs --script OR (--netlist and --sdc)")],
                    verb=self.verb, design=req.design,
                )
            res = run_pnr_flow(req.design, str(netlist), str(sdc), out_dir,
                               timeout=req.timeout_s or 1800)
            success = bool(res.get("success"))
        tool_ok = success or not _is_infra_failure(
            str(res.get("stderr", "")) + str(res.get("error", "")))
        checks = [CheckResult("pnr", "pass" if success else "fail")]
        # Report-derived checkers: PnrReportsChecker (advisory WNS/TNS/power/area)
        # and RouteDrcChecker (BLOCKING -- a routed design OpenROAD left with
        # detailed-route DRC is not a passing PnR). Both read out_dir.
        run_dir = Path(out_dir)
        for chk in self.checkers():
            checks.append(chk.check(req, run_dir))
        artifacts = {
            k: Path(res[v]) for k, v in
            (("routed_def", "routed_def_path"), ("pnr_verilog", "pnr_verilog_path"),
             ("pwr_verilog", "pwr_verilog_path"))
            if res.get(v)
        }
        lp = res.get("log_path")
        metrics = {k: res[k] for k in
                   ("design_area_um2", "wns_ns", "tns_ns", "route_drc_violations")
                   if k in res}
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks, artifacts=artifacts,
            log_path=Path(lp) if lp else None, verb=self.verb,
            design=req.design, extra_metrics=metrics,
        )

    # ------------------------------------------------------------------
    # TCL generation (moved here from backend_helpers -- OpenROAD-specific,
    # so it belongs to the tool class). All PDK data comes from the
    # deployment's PDKConfig (cells + pnr); byte-identity vs the pre-move
    # backend_helpers output is asserted by test_pdk_tcl_golden.
    # ------------------------------------------------------------------
    def render_floorplan_tcl(self, block_name: str, utilization: int,
                             gate_count: int) -> str:
        """The floorplan section (gate-count-based die sizing to avoid the
        power-strap failure IFP-0024 on small designs)."""
        import math

        pdk = self.deployment.pdk
        site = pdk.site_name
        tracks = _tracks_tcl(pdk.pnr)
        tapcell = pdk.cells.tapcell
        tap_dist = pdk.pnr.tapcell_distance

        avg_cell_area_um2 = 10
        min_edge = 60.0
        if gate_count > 0:
            estimated_edge = math.sqrt(gate_count * avg_cell_area_um2 / (utilization / 100.0)) * 2.0
            min_edge = max(60.0, estimated_edge)

        needs_explicit_die = gate_count > 0 and gate_count < 500

        if needs_explicit_die:
            core_margin = 2.5
            core_edge = min_edge - 2 * core_margin
            floorplan = (
                f'# Small design ({gate_count} gates) -- use explicit die area\n'
                f'# to ensure enough space for power straps (avoid IFP-0024).\n'
                f'initialize_floorplan \\\n'
                f'    -die_area "0 0 {min_edge:.1f} {min_edge:.1f}" \\\n'
                f'    -core_area "{core_margin} {core_margin} {core_edge:.1f} {core_edge:.1f}" \\\n'
                f'    -site {site}\n'
            )
        else:
            floorplan = (
                f'initialize_floorplan \\\n'
                f'    -utilization {utilization} \\\n'
                f'    -aspect_ratio 1.0 \\\n'
                f'    -core_space 2 \\\n'
                f'    -site {site}\n'
            )

        relaxed_util = max(utilization - 10, 15)

        return (
            f'{floorplan}\n'
            f'{tracks}\n'
            f'place_pins -hor_layers met3 -ver_layers met2\n\n'
            f'tapcell \\\n'
            f'    -distance {tap_dist} \\\n'
            f'    -tapcell_master {tapcell}\n\n'
            f'set die_area [ord::get_die_area]\n'
            f'puts "Die area: $die_area"\n\n'
            f'# Post-init die size check\n'
            f'set die_w [expr {{[lindex $die_area 2] - [lindex $die_area 0]}}]\n'
            f'set die_h [expr {{[lindex $die_area 3] - [lindex $die_area 1]}}]\n'
            f'if {{$die_w < 50.0 || $die_h < 50.0}} {{\n'
            f'    puts "WARNING: Die ${{die_w}} x ${{die_h}} um too small for PDN."\n'
            f'    initialize_floorplan -die_area "0 0 {min_edge:.1f} {min_edge:.1f}" '
            f'-core_area "2.5 2.5 {min_edge - 2.5:.1f} {min_edge - 2.5:.1f}" -site {site}\n'
            f'    {tracks}'
            f'    place_pins -hor_layers met3 -ver_layers met2\n'
            f'    set die_area [ord::get_die_area]\n'
            f'    puts "Resized die area: $die_area"\n'
            f'}}\n\n'
            f'# Post-floorplan utilization sanity check\n'
            f'set fp_die_w [expr {{[lindex $die_area 2] - [lindex $die_area 0]}}]\n'
            f'set fp_die_h [expr {{[lindex $die_area 3] - [lindex $die_area 1]}}]\n'
            f'set fp_core_area [expr {{$fp_die_w * $fp_die_h}}]\n'
            f'set fp_cell_count [llength [get_cells *]]\n'
            f'set fp_est_cell_area [expr {{$fp_cell_count * 10.0}}]\n'
            f'if {{$fp_core_area > 0}} {{\n'
            f'    set fp_actual_util [expr {{$fp_est_cell_area / $fp_core_area * 100.0}}]\n'
            f'    puts "Floorplan check: die ${{fp_die_w}}x${{fp_die_h}} um, '
            f'target util: {utilization}%, actual: ${{fp_actual_util}}%"\n'
            f'    if {{$fp_actual_util > {utilization * 1.5}}} {{\n'
            f'        puts "WARNING: utilization ${{fp_actual_util}}% exceeds 1.5x target '
            f'({utilization}%) -- re-floorplanning with {relaxed_util}%"\n'
            f'        initialize_floorplan -utilization {relaxed_util} '
            f'-aspect_ratio 1.0 -core_space 2 -site {site}\n'
            f'        {tracks}'
            f'        place_pins -hor_layers met3 -ver_layers met2\n'
            f'        set die_area [ord::get_die_area]\n'
            f'        puts "Re-floorplanned die area: $die_area"\n'
            f'    }}\n'
            f'}}\n'
        )

    def render_pnr_tcl(self, block_name: str, actual_module: str,
                       abs_netlist: str, abs_sdc: str, utilization: int = 45,
                       density: float = 0.6, gate_count: int = 0) -> str:
        """The full OpenROAD PnR script for a flat block, byte-identical to the
        pre-move ``backend_helpers.generate_pnr_tcl``."""
        pdk = self.deployment.pdk
        tech_lef, cell_lef, liberty = (self.deployment.tech_lef,
                                       self.deployment.cell_lef,
                                       self.deployment.liberty)
        pdn = pdk.pnr.pdn
        dont_use = _dont_use_tcl(pdk.cells)
        clkbufs = _clkbuf_list_tcl(pdk.cells)
        clkbuf_root = pdk.cells.clkbuf_root
        fillers = _fillers_tcl(pdk.cells)
        max_fanout = pdk.pnr.max_fanout
        pad = pdk.pnr.global_place_pad
        wire_sig = pdk.pnr.wire_rc_signal_layer
        wire_clk = pdk.pnr.wire_rc_clock_layer
        route_sig = pdk.pnr.routing_signal
        route_clk = pdk.pnr.routing_clock

        return f"""# Auto-generated PnR flow for {block_name} (Sky130 HD)
# Generated by coresmith backend_helpers.generate_pnr_tcl

set script_dir [file dirname [file normalize [info script]]]

# ----- PDK paths (absolute) -----
set tech_lef   "{tech_lef}"
set cell_lef   "{cell_lef}"
set liberty    "{liberty}"

# ----- Design paths (absolute) -----
set netlist    "{abs_netlist}"
set sdc_file   "{abs_sdc}"
set out_dir    "$script_dir"

# =====================================================================
# 1. READ DESIGN
# =====================================================================
puts "========== 1. Reading design =========="

read_lef $tech_lef
read_lef $cell_lef
read_liberty $liberty
read_verilog $netlist
link_design {actual_module}
read_sdc $sdc_file

# Fix DRT-0305: Yosys constant nets (zero_, one_) typed as GROUND/POWER
# are not routable by TritonRoute. Connect them to the power grid so they
# become special nets handled by the PDN, not by the signal router.
catch {{
    add_global_connection -net VGND -inst_pattern ".*" -pin_pattern "zero_" -ground
}}
catch {{
    add_global_connection -net VPWR -inst_pattern ".*" -pin_pattern "one_" -power
}}

puts "Design linked. Cell count: [llength [get_cells *]]"

# =====================================================================
# 2. FLOORPLAN
# =====================================================================
puts "\\n========== 2. Floorplan =========="

{self.render_floorplan_tcl(block_name, utilization, gate_count)}

# =====================================================================
# 3. POWER DISTRIBUTION NETWORK (PDN)
# =====================================================================
puts "\\n========== 3. Power grid =========="

add_global_connection -net VPWR -pin_pattern "VPWR" -power
add_global_connection -net VGND -pin_pattern "VGND" -ground
add_global_connection -net VPWR -pin_pattern "VPB" -power
add_global_connection -net VGND -pin_pattern "VNB" -ground

global_connect

set_voltage_domain -name CORE -power VPWR -ground VGND

define_pdn_grid -name stdcell_grid \\
    -starts_with POWER \\
    -voltage_domain CORE \\
    -pins {pdn['grid_pins']}

add_pdn_stripe -grid stdcell_grid -layer {pdn['followpins_layer']} -width {pdn['followpins_width']} -followpins -starts_with POWER
add_pdn_stripe -grid stdcell_grid -layer {pdn['stripe_layer']} -width {pdn['stripe_width']} -pitch {pdn['stripe_pitch']} -offset {pdn['stripe_offset']} -starts_with POWER
add_pdn_connect -grid stdcell_grid -layers {{{pdn['connect_layers']}}}

pdngen

puts "PDN generated."

# =====================================================================
# 4. GLOBAL PLACEMENT
# =====================================================================
puts "\\n========== 4. Global Placement =========="

global_placement -density {density} -pad_left {pad} -pad_right {pad}

puts "Global placement done."

# =====================================================================
# 5. DETAILED PLACEMENT
# =====================================================================
puts "\\n========== 5. Detailed Placement =========="

detailed_placement
check_placement -verbose

# NO filler insertion here -- fillers are inserted after CTS to avoid
# DPL-0036 failures when CTS buffers need placement sites occupied by
# pre-CTS fillers.

puts "Detailed placement done (fillers deferred until after CTS)."

# =====================================================================
# 6. SET WIRE RC (needed for CTS, timing repair, and STA)
# =====================================================================
puts "\\n========== 6. Set wire RC parasitics =========="

set_wire_rc -signal -layer {wire_sig}
set_wire_rc -clock  -layer {wire_clk}

puts "Wire RC set: signal={wire_sig}, clock={wire_clk}"

# =====================================================================
# 6b. PRE-CTS DESIGN REPAIR (buffer high-fanout nets, resize weak drivers)
# =====================================================================
# A ~200-fanout net left on a single min-size driver measures ~12 ns of pure
# cell delay pre-repair (live run: reset net, WNS -1.92 ns); repair_design
# with a fanout cap moved it to WNS 0.00 at 50 MHz. Also exclude the sky130
# probe/lpflow cells: the resizer otherwise picks them and they break LVS
# and skew timing.
puts "\n========== 6b. Pre-CTS repair_design =========="

set_dont_use {{{dont_use}}}
estimate_parasitics -placement
set_max_fanout {max_fanout} [current_design]
repair_design
detailed_placement
check_placement -verbose

puts "Pre-CTS repair_design done."

# =====================================================================
# 7. CLOCK TREE SYNTHESIS
# =====================================================================
puts "\\n========== 7. Clock Tree Synthesis =========="

clock_tree_synthesis \\
    -buf_list {{{clkbufs}}} \\
    -root_buf {clkbuf_root} \\
    -sink_clustering_enable

set_propagated_clock [all_clocks]

repair_clock_nets

remove_fillers
detailed_placement
filler_placement -prefix FILLER {{{fillers}}}

puts "CTS done."

# =====================================================================
# 8. TIMING REPAIR (post-CTS)
# =====================================================================
puts "\\n========== 8. Post-CTS Timing Repair =========="

estimate_parasitics -placement

repair_timing -setup
repair_timing -hold

remove_fillers
detailed_placement
check_placement -verbose
filler_placement -prefix FILLER {{{fillers}}}

puts "Post-CTS repair done."

# =====================================================================
# 9. GLOBAL ROUTING
# =====================================================================
puts "\\n========== 9. Global Routing =========="

set_routing_layers -signal {route_sig} -clock {route_clk}

global_route -guide_file "$out_dir/route_guide.guide" \\
    -congestion_iterations 50

puts "Global routing done."

# =====================================================================
# 10. DETAILED ROUTING
# =====================================================================
puts "\\n========== 10. Detailed Routing =========="

# Fix DRT-0305: Yosys/OpenROAD may create constant nets (zero_, one_)
# typed as GROUND/POWER that TritonRoute refuses to route as signal nets.
# Reclassify any non-special GROUND/POWER nets to SIGNAL before routing.
set block [ord::get_db_block]
foreach net [$block getNets] {{
    set sig_type [$net getSigType]
    set special [$net isSpecial]
    if {{($sig_type == "GROUND" || $sig_type == "POWER") && !$special}} {{
        set net_name [$net getName]
        if {{$net_name ne "VPWR" && $net_name ne "VGND" && $net_name ne "VPB" && $net_name ne "VNB"}} {{
            puts "Reclassifying net '$net_name' ($sig_type, special=$special) to SIGNAL"
            $net setSigType SIGNAL
        }}
    }}
}}

detailed_route \\
    -output_drc "$out_dir/route_drc.rpt" \\
    -verbose 1

puts "Detailed routing done."

# =====================================================================
# 11. SPEF PARASITIC ESTIMATION (in-flow)
# =====================================================================
puts "\\n========== 11. SPEF Parasitic Estimation =========="

estimate_parasitics -global_routing

# write_spef may produce empty file if estimate_parasitics didn't populate
# the RCX data store (expected -- use standalone RCX for accurate SPEF)
catch {{write_spef "$out_dir/{block_name}.spef"}}

puts "SPEF estimation done (use standalone RCX for accurate extraction)."

# =====================================================================
# 12. REPORTS (post-route STA)
# =====================================================================
puts "\\n========== 12. Reports =========="

report_checks -path_delay max -format full_clock_expanded > "$out_dir/timing_setup.rpt"
report_checks -path_delay min -format full_clock_expanded > "$out_dir/timing_hold.rpt"
report_tns > "$out_dir/timing_tns.rpt"
report_wns > "$out_dir/timing_wns.rpt"
report_power > "$out_dir/power.rpt"
puts "Reports written to $out_dir"

# Print key metrics to stdout for parsing
puts "\\n========== SUMMARY =========="
report_design_area
report_wns
report_tns
report_power

# =====================================================================
# 13. METAL DENSITY FILL (Efabless shuttle requirement)
# =====================================================================
puts "\\n========== 13. Metal Density Fill =========="

density_fill -rules $tech_lef

puts "Density fill done."

# =====================================================================
# 14. WRITE OUTPUTS
# =====================================================================
puts "\\n========== 14. Writing outputs =========="

write_def "$out_dir/{block_name}_routed.def"
write_verilog "$out_dir/{block_name}_pnr.v"
write_verilog -include_pwr_gnd "$out_dir/{block_name}_pwr.v"

puts "\\n========== FLOW COMPLETE =========="
puts "DEF:              $out_dir/{block_name}_routed.def"
puts "Verilog:          $out_dir/{block_name}_pnr.v"
puts "Power Verilog:    $out_dir/{block_name}_pwr.v"
puts "SPEF:             $out_dir/{block_name}.spef"

exit
"""

    def render_wrapper_pnr_tcl(self, wrapper_netlist_abs: str, sdc_path: str,
                               top_module: str, die_width_um: float,
                               die_height_um: float, core_margin_um: float) -> str:
        """OpenFrame wrapper-level PnR (flat, fixed die). PDK data (tracks,
        PDN, CTS bufs, fillers, site) from the deployment; the die geometry is
        an OpenFrame parameter passed by the caller."""
        pdk = self.deployment.pdk
        site = pdk.site_name
        tracks = _tracks_tcl(pdk.pnr).rstrip("\n")
        pdn = pdk.pnr.pdn
        clkbufs = _clkbuf_list_tcl(pdk.cells)
        clkbuf_root = pdk.cells.clkbuf_root
        fillers = _fillers_tcl(pdk.cells)
        wire_sig = pdk.pnr.wire_rc_signal_layer
        wire_clk = pdk.pnr.wire_rc_clock_layer
        route_sig = pdk.pnr.routing_signal
        route_clk = pdk.pnr.routing_clock
        tech_lef, cell_lef, liberty = (self.deployment.tech_lef,
                                       self.deployment.cell_lef,
                                       self.deployment.liberty)
        return f"""# Auto-generated wrapper-level PnR for OpenFrame (Sky130)
# Generated by coresmith tapeout_helpers

set script_dir [file dirname [file normalize [info script]]]
set out_dir "$script_dir"

# ---- Read PDK ----
read_lef "{tech_lef}"
read_lef "{cell_lef}"
read_liberty "{liberty}"

# ---- Read wrapper netlist (flattened, includes block logic) ----
read_verilog "{wrapper_netlist_abs}"
link_design {top_module}
read_sdc "{sdc_path}"

# ---- Fixed OpenFrame die ----
initialize_floorplan \\
    -die_area "0 0 {die_width_um:.1f} {die_height_um:.1f}" \\
    -core_area "{core_margin_um:.1f} {core_margin_um:.1f} \\
                {die_width_um - core_margin_um:.1f} \\
                {die_height_um - core_margin_um:.1f}" \\
    -site {site}

# Routing tracks
{tracks}

place_pins -hor_layers met3 -ver_layers met2

# ---- PDN (wrapper-level: met4 + met5) ----
add_global_connection -net VPWR -pin_pattern "VPWR" -power
add_global_connection -net VGND -pin_pattern "VGND" -ground
add_global_connection -net VPWR -pin_pattern "VPB" -power
add_global_connection -net VGND -pin_pattern "VNB" -ground
global_connect

set_voltage_domain -name CORE -power VPWR -ground VGND
define_pdn_grid -name wrapper_grid -starts_with POWER -voltage_domain CORE -pins {{met4 met5}}
add_pdn_stripe -grid wrapper_grid -layer {pdn['followpins_layer']} -width {pdn['followpins_width']} -followpins -starts_with POWER
add_pdn_stripe -grid wrapper_grid -layer met4 -width {pdn['stripe_width']} -pitch {pdn['stripe_pitch']} -offset {pdn['stripe_offset']} -starts_with POWER
add_pdn_stripe -grid wrapper_grid -layer met5 -width {pdn['stripe_width']} -pitch {pdn['stripe_pitch']} -offset {pdn['stripe_offset']} -starts_with POWER
add_pdn_connect -grid wrapper_grid -layers {{met1 met4}}
add_pdn_connect -grid wrapper_grid -layers {{met4 met5}}
pdngen

# ---- Standard cell placement and routing ----
global_placement -density 0.3 -pad_left 2 -pad_right 2
detailed_placement
check_placement -verbose

# Filler / decap cells for continuous n-well and power rail
filler_placement -prefix FILLER {{{fillers}}}

set_wire_rc -signal -layer {wire_sig}
set_wire_rc -clock  -layer {wire_clk}

clock_tree_synthesis \\
    -buf_list {{{clkbufs}}} \\
    -root_buf {clkbuf_root} \\
    -sink_clustering_enable
set_propagated_clock [all_clocks]
repair_clock_nets
remove_fillers
detailed_placement
filler_placement -prefix FILLER {{{fillers}}}

set_routing_layers -signal {route_sig} -clock {route_clk}
global_route -congestion_iterations 50
detailed_route -output_drc "$out_dir/wrapper_route_drc.rpt" -verbose 1

# ---- Reports ----
report_checks -path_delay max > "$out_dir/wrapper_timing_setup.rpt"
report_checks -path_delay min > "$out_dir/wrapper_timing_hold.rpt"
report_wns > "$out_dir/wrapper_wns.rpt"
report_tns > "$out_dir/wrapper_tns.rpt"
report_power > "$out_dir/wrapper_power.rpt"

puts "\\n========== WRAPPER SUMMARY =========="
report_design_area
report_wns
report_tns

# ---- Write outputs ----
write_def "$out_dir/openframe_project_wrapper_routed.def"
write_verilog "$out_dir/openframe_project_wrapper_pnr.v"
write_verilog -include_pwr_gnd "$out_dir/openframe_project_wrapper_pwr.v"

puts "\\n========== WRAPPER PNR COMPLETE =========="

exit
"""

    def checkers(self) -> list[Checker]:
        return [PnrReportsChecker(), RouteDrcChecker()]

    def reference_script(self) -> Path | None:
        ref = _DATA_DIR / "pnr_reference.tcl"
        return ref if ref.exists() else None

    def prompt_notes(self) -> str:
        return (
            "OpenROAD place-and-route on the sky130 HD PDK. Start from the "
            "reference TCL (`tool emit-script run_pnr`), which already has the "
            "full flow (read design, floorplan, PDN, placement, CTS, timing "
            "repair, routing, reports, output). Sky130-specific rules that MUST "
            "hold when you edit it:\n"
            "- Power grid MUST use met1 followpins -- sky130 HD cells require it.\n"
            "- Do NOT insert filler cells before CTS (CTS buffers need free "
            "sites); ALWAYS `remove_fillers` before any post-CTS "
            "`detailed_placement`; insert fillers only after that passes.\n"
            "- Die area must be >= 60 um on each side.\n"
            "- NEVER drop the SRAM macro reads/placement (`read_lef` of the "
            "macro LEFs, `place_macro`) to work around an error -- a layout with "
            "bound memories physically absent is a HARD FAILURE, not a success."
        )


class RunStaOpenroad(EdaTool):
    verb: ClassVar[str] = "run_sta"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.backend_helpers import run_openroad

        script = req.input("script")
        if script is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail",
                                    details="run_sta needs --script (STA tcl)")],
                verb=self.verb, design=req.design,
            )
        res = run_openroad(str(script), req.design, "sta",
                           timeout=req.timeout_s or 900)
        success = bool(res.get("success"))
        tool_ok = success or not _is_infra_failure(str(res.get("stderr", "")))
        checks = [CheckResult("sta", "pass" if success else "fail")]
        run_dir = Path(req.out_dir) if req.out_dir else Path(script).parent
        for chk in self.checkers():
            checks.append(chk.check(req, run_dir))
        lp = res.get("log_path")
        metrics = {k: res[k] for k in ("wns_ns", "tns_ns") if k in res}
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks,
            log_path=Path(lp) if lp else None, verb=self.verb,
            design=req.design, extra_metrics=metrics,
        )

    def render_rcx_tcl(self, block_name: str, abs_def: str,
                       abs_sdc: str) -> str:
        """OpenRCX SPEF extraction script (byte-identical to the pre-move
        ``backend_helpers.generate_rcx_tcl``). Via resistances are sky130
        calibration data kept inline in this sky130-owned generator (plan
        allows long-tail data as a module table, not schematized)."""
        tech_lef, cell_lef, liberty = (self.deployment.tech_lef,
                                       self.deployment.cell_lef,
                                       self.deployment.liberty)
        rcx_rules_path = str(self.deployment.paths.rcx_rules)
        return f"""# Auto-generated OpenRCX SPEF extraction for {block_name} (Sky130)
# Generated by coresmith backend_helpers.generate_rcx_tcl

set script_dir [file dirname [file normalize [info script]]]

set tech_lef   "{tech_lef}"
set cell_lef   "{cell_lef}"
set liberty    "{liberty}"
set sdc_file   "{abs_sdc}"
set def_file   "{abs_def}"
set rcx_rules  "{rcx_rules_path}"
set out_dir    "$script_dir"

puts "========== RCX SPEF Extraction: {block_name} =========="

# 1. Read design from DEF
read_lef $tech_lef
read_lef $cell_lef
read_liberty $liberty
read_def $def_file
read_sdc $sdc_file

set_propagated_clock [all_clocks]

# 2. Set via resistances (Sky130 calibration from OpenROAD-flow-scripts)
set tech [ord::get_db_tech]
[$tech findLayer mcon] setResistance 9.249146
[$tech findLayer via]  setResistance 4.5
[$tech findLayer via2] setResistance 3.368786
[$tech findLayer via3] setResistance 0.376635
[$tech findLayer via4] setResistance 0.00580

# 3. Run OpenRCX extraction
puts "Running OpenRCX extraction..."
define_process_corner -ext_model_index 0 X
extract_parasitics -ext_model_file $rcx_rules

# 4. Write SPEF
puts "Writing SPEF..."
write_spef "$out_dir/{block_name}.spef"

# 5. Write power-aware Verilog for LVS
write_verilog -include_pwr_gnd "$out_dir/{block_name}_pwr.v"

# 6. Post-extraction STA
puts "\\n========== Post-extraction STA =========="
report_checks -path_delay max -format full_clock_expanded
report_checks -path_delay min -format full_clock_expanded
report_tns
report_wns
report_power

puts "\\nSPEF: $out_dir/{block_name}.spef"
puts "Power Verilog: $out_dir/{block_name}_pwr.v"
puts "Done."

exit
"""

    def checkers(self) -> list[Checker]:
        return [StaChecker()]

    def prompt_notes(self) -> str:
        return (
            "OpenROAD/OpenSTA on the sky130 HD liberty: "
            "read_liberty -> read_verilog -> link_design -> read_sdc -> "
            "report_checks / report_wns / report_tns. Report WNS/TNS in ns."
        )


class RunDrcMagic(EdaTool):
    verb: ClassVar[str] = "run_drc"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.backend_helpers import run_drc_flow, run_magic

        out_dir = str(req.out_dir or (PROJECT_ROOT / "pnr" / "output" / req.design))
        script = req.input("script")
        if script is not None:
            res = run_magic(str(script), req.design, "drc",
                            timeout=req.timeout_s or 600)
            success = bool(res.get("success"))
            clean = res.get("drc_count", 1) == 0
        else:
            layout = req.input("gds")  # routed layout (DEF/GDS) to check
            if layout is None:
                return ToolResult.from_checks(
                    tool_ok=False,
                    checks=[CheckResult("inputs", "fail",
                                        details="run_drc needs --script OR --gds")],
                    verb=self.verb, design=req.design,
                )
            res = run_drc_flow(req.design, str(layout), out_dir,
                               timeout=req.timeout_s or 600)
            success = bool(res.get("success"))
            clean = bool(res.get("clean"))
        tool_ok = success or not _is_infra_failure(str(res.get("stderr", "")))
        checks = [CheckResult("drc", "pass" if clean else "fail",
                              metrics={"violations": res.get("violation_count",
                                                             res.get("drc_count", 0))})]
        # Honest report-derived DRC verdict (BLOCKING, fail-closed): catches
        # report-rects the stdout count missed and never renders a missing
        # report as clean (drc_verdict.classify_drc three-state).
        run_dir = Path(out_dir)
        for chk in self.checkers():
            checks.append(chk.check(req, run_dir))
        artifacts = {k: Path(res[v]) for k, v in
                     (("gds", "gds_path"), ("spice", "spice_path")) if res.get(v)}
        lp = res.get("log_path")
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks, artifacts=artifacts,
            log_path=Path(lp) if lp else None, verb=self.verb, design=req.design,
        )

    def checkers(self) -> list[Checker]:
        return [MagicDrcChecker()]

    def prompt_notes(self) -> str:
        return (
            "Magic VLSI DRC + SPICE extraction on the sky130 HD PDK. Author a "
            "Magic TCL script; the CLI runs it under the correct magicrc. "
            "Sky130-specific requirements:\n"
            "- Read the tech + cell LEFs FIRST, BEFORE `def read` -- without the "
            "tech LEF, Magic cannot map the DEF's routing vias and every net "
            "becomes a dangling single-pin node (LVS then reports a huge "
            "net_delta no setup can reconcile). Read each macro LEF here too.\n"
            "- Black-box each hard macro before extraction (`extract halt "
            "<cell>`) so Magic does not descend into the intractable "
            "transistor-level macro layout.\n"
            "- Run DRC HIERARCHICALLY -- do NOT flatten for the gating count "
            "(flattening reports thousands of PDK-derivation sliver artifacts). "
            "`drc check; drc catchup` then `drc listall why <report>`.\n"
            "- Extract LVS SPICE CONNECTIVITY-ONLY (no parasitics): "
            "`extract do local; extract no capacitance; extract no coupling; "
            "extract no resistance; extract all; ext2spice lvs; "
            "ext2spice cthresh infinite; ext2spice rthresh infinite`.\n"
            "- The signoff GDS is written by the deployment (KLayout streamout "
            "from the DEF, Magic `gds write` fallback) -- you do not need to "
            "script the DEF-to-GDS conversion yourself."
        )


class RunLvsNetgen(EdaTool):
    verb: ClassVar[str] = "run_lvs"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.backend_helpers import run_lvs_flow

        # Agent-authored netgen script path (the abstraction-aligned readnet
        # flow); no script -> the deterministic spice-vs-netlist helper.
        script = req.input("script")
        if script is not None:
            return self._run_script(req, script)

        spice, netlist = req.input("spice"), req.input("netlist")
        if spice is None or netlist is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail",
                                    details="run_lvs needs --script OR (--spice and --netlist)")],
                verb=self.verb, design=req.design,
            )
        out_dir = str(req.out_dir or (PROJECT_ROOT / "pnr" / "output" / req.design))
        res = run_lvs_flow(req.design, str(spice), str(netlist), out_dir,
                           timeout=req.timeout_s or 600)
        match = bool(res.get("match"))
        tool_ok = ("match" in res) or not _is_infra_failure(str(res.get("stderr", "")))
        checks = [CheckResult("lvs", "pass" if match else "fail",
                              metrics={"device_delta": res.get("device_delta", 0),
                                       "net_delta": res.get("net_delta", 0)})]
        rp = res.get("report_path")
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks,
            artifacts={"report": Path(rp)} if rp else {},
            log_path=Path(res["log_path"]) if res.get("log_path") else None,
            verb=self.verb, design=req.design,
        )

    def _run_script(self, req: ToolRequest, script: Path) -> ToolResult:
        """Run a caller-authored ``netgen -batch source <script>`` and derive the
        verdict from the report (LvsMatchChecker reconciles benign pins)."""
        out_dir = Path(req.out_dir) if req.out_dir else (
            PROJECT_ROOT / "pnr" / "output" / (req.design or "lvs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rc, stdout, stderr, infra = _run_cmd(
            [self.deployment.netgen_bin, "-batch", "source", str(script)],
            timeout=req.timeout_s or 900, cwd=str(PROJECT_ROOT))
        log = out_dir / f"{req.design or 'lvs'}_netgen.log"
        log.write_text(stdout + "\n" + stderr)
        tool_ok = not (infra or _is_infra_failure(stderr))
        checks = [LvsMatchChecker().check(req, out_dir)]
        rpt = sorted(out_dir.glob("*lvs*.rpt"))
        return ToolResult.from_checks(
            tool_ok=tool_ok, checks=checks,
            artifacts={"report": rpt[0]} if rpt else {},
            log_path=log, verb=self.verb, design=req.design,
        )

    def checkers(self) -> list[Checker]:
        # The inline verdict above already reconciles benign pins WITH the
        # reference power-Verilog (richer than a report-only standalone check),
        # so it stays the source of truth; LvsMatchChecker is exposed here for
        # `tool list` + standalone use over a report file.
        return [LvsMatchChecker()]

    def prompt_notes(self) -> str:
        return (
            "Netgen LVS on the sky130 HD PDK: compare the Magic-extracted SPICE "
            "against the power-aware PnR Verilog. Sky130/openframe specifics:\n"
            "- A cell-library SPICE plus the layout netlist must each be read as "
            "SEPARATE `readnet` calls (netgen treats a space-joined pair as one "
            "filename); verify the cell-library subckts loaded.\n"
            "- Power-pin (VPWR/VGND/VPB/VNB) and openframe GPIO tie mismatches "
            "are the expected benign artifact -- a `Final result: Circuits match "
            "uniquely` after benign reconciliation is a pass; keep the report so "
            "the checker can reconcile it."
        )


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------
class Sky130Deployment(Deployment):
    name: ClassVar[str] = "sky130"

    def __init__(self) -> None:
        self._pdk_root = _current_pdk_root()
        self.paths = _resolve_paths(self._pdk_root)
        # Backend tool binaries (env -> config.yaml -> nix wrapper -> PATH).
        self.openroad_bin = _resolve_tool("openroad_binary", "scripts/openroad-nix.sh")
        self.magic_bin = _resolve_tool("magic_binary", "scripts/magic-nix.sh")
        self.netgen_bin = _resolve_tool("netgen_binary", "scripts/netgen-nix.sh")
        self.klayout_bin = _resolve_tool("klayout_binary", "scripts/klayout-nix.sh")
        self._tools: dict[str, EdaTool] = {
            "run_synth": RunSynthYosys(self),
            "run_pnr": RunPnrOpenroad(self),
            "run_sta": RunStaOpenroad(self),
            "run_drc": RunDrcMagic(self),
            "run_lvs": RunLvsNetgen(self),
            "run_lint": RunLintVerilator(self),
        }
        self._pdk: PDKConfig | None = None

    # -- PDK path accessors (also exposed as module-level constants below) ----
    @property
    def tech_lef(self) -> Path:
        return self.paths.tech_lef

    @property
    def cell_lef(self) -> Path:
        return self.paths.cell_lef

    @property
    def liberty(self) -> Path:
        return self.paths.liberty

    def resolve_yosys(self) -> str:
        return _resolve_yosys()

    @property
    def pdk(self) -> PDKConfig:
        if self._pdk is None:
            self._pdk = PDKConfig.from_yaml(str(_CONFIG_YAML),
                                            pdk_root=str(self._pdk_root))
        return self._pdk

    def tools(self) -> dict[str, EdaTool]:
        return self._tools

    def sim_models(self) -> list[Path]:
        gl = self.paths.pdk_path / "libs.ref" / _STD_CELL / "verilog"
        return [gl] if gl.is_dir() else []

    def data_dir(self) -> Path | None:
        return _DATA_DIR if _DATA_DIR.is_dir() else None

    # -- data_dir payload accessors (the deployment OWNS its TCL/data files) --
    def pnr_reference_tcl(self) -> Path:
        """The PnR reference template an agent adapts (``tool emit-script``)."""
        return _DATA_DIR / "pnr_reference.tcl"

    def load_template(self, name: str) -> str:
        """Load a data_dir Tcl template by name (without extension)."""
        return (_DATA_DIR / f"{name}.tcl").read_text()

    def filler_cells(self) -> list[str]:
        """Ordered filler/decap cells: the data_dir file if present, else the
        PDKConfig ``cells.fillers`` list."""
        f = _DATA_DIR / "filler_cells.txt"
        if f.is_file():
            return [ln.strip() for ln in f.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]
        return list(self.pdk.cells.fillers)

    def def2gds(self, def_path: str, out_dir: str, block_name: str,
                timeout: int = 600) -> dict:
        """Write a signoff GDS from a routed DEF -- KLayout streamout first
        (OpenLane-standard), Magic ``gds write`` fallback. This is DEPLOYMENT
        LOGIC (formerly agent-prompt guidance): the caller no longer scripts the
        DEF-to-GDS conversion.

        Returns ``{ok, gds_path, method, log}``; fail-closed -- if neither tool
        produces a GDS, ``ok`` is False and ``gds_path`` is None (never a
        fabricated path).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{block_name}.gds"
        logs: list[str] = []

        # 1. KLayout streamout via the PDK's own klayout tech (.lyt), the
        #    OpenLane-standard path. Skipped honestly when the binary or the
        #    tech file is absent.
        lyt = (self.paths.pdk_path / "libs.tech" / "klayout" / "tech"
               / f"{self.paths.variant}.lyt")
        if lyt.is_file():
            rb = out / "_def2gds.rb"
            rb.write_text(
                f'lm = RBA::LayoutMetaInfo\n'
                f'ly = RBA::Layout::new\n'
                f'ly.read("{def_path}", '
                f'RBA::LoadLayoutOptions::new)\n'
                f'ly.write("{target}")\n')
            rc, so, se, infra = _run_cmd(
                [self.klayout_bin, "-b", "-r", str(rb)], timeout=timeout,
                cwd=str(PROJECT_ROOT))
            logs.append(f"[klayout] rc={rc} {infra} {se[-200:]}")
            if rc == 0 and target.is_file() and target.stat().st_size > 0:
                return {"ok": True, "gds_path": str(target),
                        "method": "klayout", "log": "\n".join(logs)}

        # 2. Magic `gds write` fallback (the def2gds.tcl data-dir template).
        try:
            tmpl = self.load_template("def2gds")
        except OSError as exc:
            logs.append(f"[magic] no def2gds template: {exc}")
            return {"ok": False, "gds_path": None, "method": None,
                    "log": "\n".join(logs)}
        script = (tmpl
                  .replace("$TECH_LEF", str(self.paths.tech_lef))
                  .replace("$CELL_LEF", str(self.paths.cell_lef))
                  .replace("$CELL_GDS", str(self.paths.cell_gds))
                  .replace("$DEF_FILE", def_path)
                  .replace("$BLOCK_NAME", block_name)
                  .replace("$OUT_DIR", str(out)))
        tcl = out / "_def2gds.tcl"
        tcl.write_text(script)
        rc, so, se, infra = _run_cmd(
            [self.magic_bin, "-dnull", "-noconsole", "-rcfile",
             str(self.paths.magic_rc), str(tcl)],
            timeout=timeout, cwd=str(PROJECT_ROOT))
        logs.append(f"[magic] rc={rc} {infra} {se[-200:]}")
        if target.is_file() and target.stat().st_size > 0:
            return {"ok": True, "gds_path": str(target),
                    "method": "magic", "log": "\n".join(logs)}
        return {"ok": False, "gds_path": None, "method": None,
                "log": "\n".join(logs)}

    def prompt_context(self) -> dict[str, str]:
        ctx = super().prompt_context()
        ctx["std_cell_library"] = _STD_CELL
        ctx["pdk_variant"] = self.paths.variant
        # Cell SPICE library path, so an LVS script can `readnet` the reference
        # cell definitions without a hardcoded cell-library token in the prompt.
        ctx["cell_spice"] = str(self.paths.cell_spice)
        return ctx


DEPLOYMENT = Sky130Deployment()

# ---------------------------------------------------------------------------
# Module-level constants -- the values ``backend_helpers`` re-exports so every
# existing call site (`from ...backend_helpers import LIBERTY`, etc.) keeps its
# byte-identical value. Frozen at import from the import-time PDK_ROOT, exactly
# like the old module-level constants were.
# ---------------------------------------------------------------------------
_PDK_VAR = DEPLOYMENT.paths.variant
_PDK_PATH = DEPLOYMENT.paths.pdk_path
TECH_LEF = DEPLOYMENT.paths.tech_lef
CELL_LEF = DEPLOYMENT.paths.cell_lef
LIBERTY = DEPLOYMENT.paths.liberty
CELL_GDS = DEPLOYMENT.paths.cell_gds
CELL_SPICE = DEPLOYMENT.paths.cell_spice
MAGIC_RC = DEPLOYMENT.paths.magic_rc
NETGEN_SETUP = DEPLOYMENT.paths.netgen_setup
RCX_RULES = DEPLOYMENT.paths.rcx_rules
OPENROAD_BIN = DEPLOYMENT.openroad_bin
MAGIC_BIN = DEPLOYMENT.magic_bin
NETGEN_BIN = DEPLOYMENT.netgen_bin
KLAYOUT_BIN = DEPLOYMENT.klayout_bin
