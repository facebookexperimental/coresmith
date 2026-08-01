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

        rtl = req.input("rtl")
        if rtl is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail", details="run_synth needs --rtl")],
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

    def checkers(self) -> list[Checker]:
        return [SynthStatChecker(), LogicDepthChecker(),
                OutputArtifactChecker("netlist_present", "netlist")]

    def prompt_notes(self) -> str:
        return (
            "yosys synth targeting sky130_fd_sc_hd; PDK-free generic mapping when "
            "no liberty is present. dfflibmap + abc -liberty for the mapped flow."
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

    def checkers(self) -> list[Checker]:
        return [PnrReportsChecker(), RouteDrcChecker()]

    def reference_script(self) -> Path | None:
        ref = _DATA_DIR / "pnr_reference.tcl"
        return ref if ref.exists() else None

    def prompt_notes(self) -> str:
        return "OpenROAD -no_init -exit <tcl>; start from the pnr_reference.tcl template."


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

    def checkers(self) -> list[Checker]:
        return [StaChecker()]

    def prompt_notes(self) -> str:
        return "OpenROAD STA: read_liberty/read_verilog/read_sdc + report_checks."


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
        return "Magic -dnull -noconsole -rcfile <magicrc> <drc.tcl>."


class RunLvsNetgen(EdaTool):
    verb: ClassVar[str] = "run_lvs"

    def run(self, req: ToolRequest) -> ToolResult:
        from orchestrator.langgraph.backend_helpers import run_lvs_flow

        spice, netlist = req.input("spice"), req.input("netlist")
        if spice is None or netlist is None:
            return ToolResult.from_checks(
                tool_ok=False,
                checks=[CheckResult("inputs", "fail",
                                    details="run_lvs needs --spice and --netlist")],
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

    def checkers(self) -> list[Checker]:
        # The inline verdict above already reconciles benign pins WITH the
        # reference power-Verilog (richer than a report-only standalone check),
        # so it stays the source of truth; LvsMatchChecker is exposed here for
        # `tool list` + standalone use over a report file.
        return [LvsMatchChecker()]

    def prompt_notes(self) -> str:
        return "Netgen -batch lvs <spice> <verilog> <netgen-setup>."


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

    def prompt_context(self) -> dict[str, str]:
        ctx = super().prompt_context()
        ctx["std_cell_library"] = _STD_CELL
        ctx["pdk_variant"] = self.paths.variant
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
