# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Chip Finish Dashboard Generator.

Generates a comprehensive HTML dashboard summarizing the complete ASIC
design flow: architecture documents, RTL, synthesis metrics,
place-and-route results, and physical verification sign-off.

Uses a Jinja2 HTML template rendered deterministically from data on disk.
No LLM is involved -- this eliminates hallucination risk entirely.

Called as the final node in the backend graph after all blocks complete.
"""

from __future__ import annotations

import html as html_module
import json
import re
from pathlib import Path
from typing import Any

from orchestrator.architecture.specialists.dashboard_doc import (
    _blocks_to_mermaid,
)


_TEMPLATE_FILE = (
    Path(__file__).resolve().parents[2]
    / "langchain"
    / "prompts"
    / "chip_finish_template.html"
)

_GDS_SIZE_THRESHOLD = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_chip_finish_dashboard(
    completed_blocks: list[dict],
    project_root: str,
    target_clock_mhz: float,
    tapeout_state: dict | None = None,
    viewer_3d_available: bool = False,
    layout_2d_png_path: str = "",
) -> str:
    """Generate the chip finish HTML dashboard via Jinja2 template.

    Reads all architecture docs, backend reports, RTL source, testbenches,
    pipeline events, and tapeout results from disk. Renders everything
    into a Jinja2 HTML template deterministically -- no LLM involved.

    Args:
        completed_blocks: List of completed block result dicts.
        project_root: Path to the project root directory.
        target_clock_mhz: Target clock frequency.
        tapeout_state: Optional tapeout graph state dict.
        viewer_3d_available: Whether a 3D GDS viewer was generated.

    Returns the raw HTML string.
    """
    from jinja2 import Environment, BaseLoader

    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.backend.chip_finish_dashboard")

    with tracer.start_as_current_span("generate_chip_finish_dashboard") as span:
        root = Path(project_root)

        primary = _pick_primary_block(completed_blocks)
        block_name = primary.get("name", "unknown") if primary else "unknown"
        span.set_attribute("block_name", block_name)

        # Find the flat top-level design name for backend reports.
        # The integration top-level is typically the longest name in
        # syn/output/ and the one used for PnR/DRC/LVS.
        top_design = _find_top_design(root, block_name, completed_blocks)

        # -- Structured architecture data ----------------------------------
        prd_data = _read_json_safe(root / ".coresmith" / "prd_spec.json") or {}
        prd_content = prd_data.get("prd", prd_data) if prd_data else {}
        block_diagram = _read_json_safe(root / ".coresmith" / "block_diagram.json") or {}
        mermaid_diagram = _blocks_to_mermaid(block_diagram)

        # -- Backend reports -----------------------------------------------
        pnr_dir = root / "syn" / "output" / top_design / "pnr"
        def_data = _parse_def_file(pnr_dir / f"{top_design}_routed.def")
        synth_report = _parse_synthesis_report(
            root / "syn" / "output" / top_design / f"{top_design}_report.txt",
        )
        critical_path = _parse_timing_report(pnr_dir / "timing_setup.rpt")
        power_data = _parse_power_report(pnr_dir / "power.rpt")
        metrics = _build_metrics(primary, pnr_dir)

        # -- RTL source files ----------------------------------------------
        rtl_files = _collect_rtl_files(root, completed_blocks)
        span.set_attribute("rtl_file_count", len(rtl_files))

        # -- Testbench files -----------------------------------------------
        tb_files = _collect_tb_files(root, completed_blocks)
        tb_groups = _collect_tb_groups(tb_files)
        span.set_attribute("tb_file_count", len(tb_files))

        # -- Pipeline timeline ---------------------------------------------
        timeline = _build_timeline_bars(root / ".coresmith" / "pipeline_events.jsonl")

        # -- Test results --------------------------------------------------
        test_results = _read_test_results(root, block_name)

        # -- Tapeout data --------------------------------------------------
        tapeout_data = _build_tapeout_data(tapeout_state, root)
        span.set_attribute("has_tapeout", bool(tapeout_data.get("has_tapeout")))

        # -- VCD waveforms -------------------------------------------------
        vcd_waveforms = _collect_vcd_waveforms(root, completed_blocks)
        vcd_template_data = []
        for wf in vcd_waveforms:
            vcd_template_data.append({
                "block_name": wf.get("block_name", "unknown"),
                "wavedrom_json": json.dumps(wf.get("wavedrom", {})),
            })

        # -- Architecture documents ----------------------------------------
        arch_docs = _collect_arch_docs(root, block_name)

        # -- GDS viewer decision -------------------------------------------
        gds_viewer = "none"
        gds_png_path = ""
        gds_3d_path = root / "chip_finish" / "3d.html"
        gds_png_candidate = root / "chip_finish" / f"{top_design}_layout.png"

        if gds_3d_path.exists():
            try:
                size = gds_3d_path.stat().st_size
                gds_viewer = "3d" if size < _GDS_SIZE_THRESHOLD else "png"
            except OSError:
                gds_viewer = "3d"

        if gds_viewer == "png" or (gds_viewer == "none" and gds_png_candidate.exists()):
            gds_viewer = "png"
            gds_png_path = f"{top_design}_layout.png"

        # -- DEF enrichment for template -----------------------------------
        def_enriched = dict(def_data)
        die = def_data.get("die_area", {})
        if die:
            def_enriched["die_width_um"] = f"{(die.get('x2', 0) - die.get('x1', 0)) / 1000:.1f}"
            def_enriched["die_height_um"] = f"{(die.get('y2', 0) - die.get('y1', 0)) / 1000:.1f}"
            def_enriched["component_count"] = len(def_data.get("components", []))

        # -- Cell distribution for template --------------------------------
        cell_dist = {"cells": [], "total": synth_report.get("total_cells", 0)}
        for c in synth_report.get("cells", [])[:8]:
            cell_dist["cells"].append({
                "name": c["type"].replace("sky130_fd_sc_hd__", ""),
                "count": c["count"],
            })

        # -- Metrics as namespace for template -----------------------------
        class MetricsNS:
            pass
        m = MetricsNS()
        for k, v in metrics.items():
            setattr(m, k, v)
        m.gate_count = (
            metrics.get("total_cells")
            or synth_report.get("total_cells")
            or (primary.get("synth_gate_count", 0) if primary else 0)
        )

        # -- Render template -----------------------------------------------
        template_text = _TEMPLATE_FILE.read_text(encoding="utf-8")
        env = Environment(loader=BaseLoader(), autoescape=False)
        template = env.from_string(template_text)

        html = template.render(
            design_name=block_name,
            target_clock_mhz=target_clock_mhz,
            prd=prd_content if isinstance(prd_content, dict) else {},
            blocks=block_diagram.get("blocks", []),
            block_diagram=block_diagram,
            mermaid_diagram=mermaid_diagram,
            metrics=m,
            timeline=timeline,
            rtl_files=rtl_files,
            tb_files=tb_files,
            tb_groups=tb_groups,
            vcd_waveforms=vcd_template_data,
            test_results=test_results,
            critical_path=critical_path,
            cell_dist=cell_dist,
            power_breakdown=power_data,
            def_data=def_enriched,
            gds_viewer=gds_viewer,
            gds_png_path=gds_png_path,
            arch_docs=arch_docs,
            tapeout=tapeout_data,
        )

        span.set_attribute("html_length", len(html))
        return html


# ---------------------------------------------------------------------------
# Block selection
# ---------------------------------------------------------------------------


def _pick_primary_block(completed_blocks: list[dict]) -> dict | None:
    """Pick the first successful block, or the first block if none passed."""
    for b in completed_blocks:
        if b.get("success"):
            return b
    return completed_blocks[0] if completed_blocks else None


def _find_top_design(
    root: Path, fallback: str, completed_blocks: list[dict],
) -> str:
    """Find the flat top-level design name for backend report paths.

    The backend synthesises/PnR's the integrated top-level design whose
    synth output directory contains the flat netlist and PnR artifacts.
    This is typically the directory in ``syn/output/`` that has a ``pnr/``
    subdirectory and the largest synth report (flat reports include all
    submodule statistics).
    """
    syn_out = root / "syn" / "output"
    if not syn_out.is_dir():
        return fallback

    candidates: list[tuple[int, int, str]] = []
    for d in syn_out.iterdir():
        if not d.is_dir():
            continue
        report = d / f"{d.name}_report.txt"
        if not report.exists():
            continue
        has_pnr = (d / "pnr").is_dir()
        try:
            size = report.stat().st_size
        except OSError:
            size = 0
        candidates.append((1 if has_pnr else 0, size, d.name))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2]
    return fallback


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _read_file_safe(path: Path, max_chars: int = 50_000) -> str:
    """Read a text file, returning empty string on any failure."""
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if len(text) > max_chars:
                return text[:max_chars] + "\n\n... [truncated] ..."
            return text
    except OSError:
        pass
    return ""


def _read_json_safe(path: Path) -> dict | None:
    """Read a JSON file, returning None on any failure."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _read_uarch_spec(root: Path, block_name: str) -> str:
    """Read the uArch spec markdown for *block_name*."""
    for parent in ("arch", ".coresmith"):
        spec = root / parent / "uarch_specs" / f"{block_name}.md"
        if spec.exists():
            try:
                return spec.read_text(encoding="utf-8")
            except OSError:
                continue
    return ""


# ---------------------------------------------------------------------------
# RTL and testbench file collectors
# ---------------------------------------------------------------------------


def _collect_rtl_files(
    root: Path, completed_blocks: list[dict],
) -> list[dict]:
    """Collect all RTL Verilog files for display in the dashboard."""
    files: list[dict] = []
    seen: set[str] = set()

    for block in completed_blocks:
        name = block.get("name", "")
        rtl_target = block.get("rtl_target", "")
        if rtl_target:
            p = root / rtl_target
        else:
            p = None
            for sub in sorted((root / "rtl").rglob(f"{name}.v")):
                p = sub
                break

        if p and p.exists() and str(p) not in seen:
            seen.add(str(p))
            try:
                content = p.read_text(encoding="utf-8")
                if len(content) > 20_000:
                    content = content[:20_000] + "\n// ... [truncated] ..."
                files.append({
                    "id": f"rtl_{name}",
                    "name": f"{name}.v",
                    "content": html_module.escape(content),
                })
            except OSError:
                pass

    if not files:
        for vf in sorted((root / "rtl").rglob("*.v")):
            if str(vf) not in seen:
                seen.add(str(vf))
                try:
                    content = vf.read_text(encoding="utf-8")
                    if len(content) > 20_000:
                        content = content[:20_000] + "\n// ... [truncated] ..."
                    stem = vf.stem
                    files.append({
                        "id": f"rtl_{stem}",
                        "name": vf.name,
                        "content": html_module.escape(content),
                    })
                except OSError:
                    pass

    return files


def _collect_tb_files(
    root: Path, completed_blocks: list[dict],
) -> list[dict]:
    """Collect cocotb testbench files filtered to completed blocks.

    Also returns flat list for backward-compat template rendering.
    Use ``_collect_tb_groups()`` for grouped output.
    """
    block_names = {b.get("name", "") for b in completed_blocks if b.get("name")}
    files: list[dict] = []
    seen: set[str] = set()

    tb_dir = root / "tb" / "cocotb"
    if tb_dir.is_dir():
        for tf in sorted(tb_dir.glob("test_*.py")):
            stem = tf.stem
            matched_block = _match_tb_to_block(stem, block_names)
            if not matched_block:
                continue
            if str(tf) not in seen:
                seen.add(str(tf))
                try:
                    content = tf.read_text(encoding="utf-8")
                    if len(content) > 30_000:
                        content = content[:30_000] + "\n# ... [truncated] ..."
                    files.append({
                        "id": f"tb_{stem}",
                        "name": tf.name,
                        "block": matched_block,
                        "content": html_module.escape(content),
                    })
                except OSError:
                    pass

    int_tb_dir = root / "tb" / "integration"
    if int_tb_dir.is_dir():
        for tf in sorted(int_tb_dir.glob("test_*.py")):
            if str(tf) not in seen:
                seen.add(str(tf))
                try:
                    content = tf.read_text(encoding="utf-8")
                    if len(content) > 30_000:
                        content = content[:30_000] + "\n# ... [truncated] ..."
                    files.append({
                        "id": f"tb_{tf.stem}",
                        "name": tf.name,
                        "block": "integration",
                        "content": html_module.escape(content),
                    })
                except OSError:
                    pass

    return files


def _match_tb_to_block(tb_stem: str, block_names: set[str]) -> str:
    """Match a testbench filename stem (e.g. ``test_adder_8bit``) to a block."""
    suffix = tb_stem.removeprefix("test_")
    if suffix in block_names:
        return suffix
    for name in block_names:
        if suffix.startswith(name) or name.startswith(suffix):
            return name
    return ""


def _collect_tb_groups(
    tb_files: list[dict],
) -> list[dict]:
    """Group flat TB file list by block name for collapsible rendering.

    Returns ``[{"block": "adder_8bit", "files": [...]}, ...]``.
    """
    groups: dict[str, list[dict]] = {}
    for tb in tb_files:
        block = tb.get("block", "other")
        groups.setdefault(block, []).append(tb)

    result = []
    for block in sorted(groups):
        result.append({"block": block, "files": groups[block]})
    return result


def _collect_arch_docs(root: Path, block_name: str) -> list[dict]:
    """Collect architecture documents for tabbed display."""
    docs = []
    doc_map = [
        ("prdDoc", "PRD", root / "arch" / "prd_spec.md"),
        ("sadDoc", "SAD", root / "arch" / "sad_spec.md"),
        ("frdDoc", "FRD", root / "arch" / "frd_spec.md"),
        ("ersDoc", "ERS", root / "arch" / "ers_spec.md"),
        ("uarchDoc", "µArch", None),
    ]

    for doc_id, label, path in doc_map:
        if path is None:
            content = _read_uarch_spec(root, block_name)
        elif path.exists():
            content = _read_file_safe(path, max_chars=30_000)
        else:
            content = "Not available."

        docs.append({
            "id": doc_id,
            "label": label,
            "content": html_module.escape(content) if content else "Not available.",
        })

    return docs


def _build_timeline_bars(events_path: Path) -> list[dict]:
    """Build timeline bar data for the summary section."""
    raw_timeline = _build_timeline(events_path)
    if not raw_timeline:
        return []

    colors = ["#6366f1", "#3b82f6", "#06b6d4", "#8b5cf6", "#22c55e", "#f59e0b"]
    bars = []
    for i, phase in enumerate(raw_timeline):
        dur = phase.get("duration_s", 0)
        if dur < 1:
            continue
        name = phase.get("phase", "unknown")
        if dur >= 60:
            label = f"{name[:4].title()} {dur / 60:.0f}m"
        else:
            label = f"{name[:4].title()} {dur:.0f}s"
        bars.append({
            "name": name,
            "duration_s": dur,
            "label": label,
            "color": colors[i % len(colors)],
        })

    return bars


# ---------------------------------------------------------------------------
# Tapeout / MPW precheck data
# ---------------------------------------------------------------------------


def _build_tapeout_data(
    tapeout_state: dict | None,
    root: Path,
) -> dict:
    """Build tapeout results data for the dashboard.

    Reads from tapeout graph state (if available) and/or from on-disk
    precheck artifacts.
    """
    data: dict = {
        "has_tapeout": False,
        "shuttle_target": "openframe",
        "precheck_pass": False,
        "precheck_checks": {},
        "wrapper_drc_clean": False,
        "wrapper_drc_violations": -1,
        "wrapper_lvs_match": False,
        "wrapper_lvs_device_delta": 0,
        "wrapper_lvs_net_delta": 0,
        "gpio_used": 0,
        "gpio_available": 44,
        "submission_dir": "",
        "submission_files": [],
    }

    if tapeout_state:
        data["has_tapeout"] = True

        precheck = tapeout_state.get("precheck_result") or {}
        data["precheck_pass"] = precheck.get("pass", False)
        data["precheck_checks"] = {
            k: v.get("pass", False)
            for k, v in precheck.get("checks", {}).items()
        }

        drc = tapeout_state.get("wrapper_drc_result") or {}
        data["wrapper_drc_clean"] = drc.get("clean", False)
        data["wrapper_drc_violations"] = drc.get("violation_count", -1)

        lvs = tapeout_state.get("wrapper_lvs_result") or {}
        data["wrapper_lvs_match"] = lvs.get("match", False)
        data["wrapper_lvs_device_delta"] = lvs.get("device_delta", 0)
        data["wrapper_lvs_net_delta"] = lvs.get("net_delta", 0)

        wrapper = tapeout_state.get("wrapper_result") or {}
        data["gpio_used"] = wrapper.get("gpio_used", 0)
        data["gpio_available"] = wrapper.get("gpio_available", 44)

        data["submission_dir"] = tapeout_state.get("submission_dir", "")

        submission = tapeout_state.get("submission_result") or {}
        raw_files = submission.get("files_copied", [])
        data["submission_files"] = _normalize_submission_files(raw_files)

    # Also try reading from on-disk artifacts if tapeout_state is missing
    sub_dir = root / "openframe_submission"
    if sub_dir.is_dir():
        if not data["has_tapeout"]:
            data["has_tapeout"] = True
            data["submission_dir"] = str(sub_dir)
        if not data["submission_files"]:
            files = []
            for subpath in sorted(sub_dir.rglob("*")):
                if subpath.is_file():
                    try:
                        rel = str(subpath.relative_to(sub_dir))
                        files.append({"path": rel, "present": True})
                    except ValueError:
                        pass
            data["submission_files"] = files[:50]

    return data


def _normalize_submission_files(raw: list) -> list[dict]:
    """Ensure submission files are dicts with ``path`` and ``present`` keys."""
    result = []
    for item in raw:
        if isinstance(item, dict):
            path = item.get("path", "") or item.get("name", "")
            result.append({
                "path": path,
                "present": item.get("present", bool(path)),
            })
        elif isinstance(item, str) and item:
            result.append({"path": item, "present": True})
    return result


# ---------------------------------------------------------------------------
# DEF parser
# ---------------------------------------------------------------------------


def _parse_def_file(def_path: Path) -> dict:
    """Parse a DEF file for components, pins, rows, and die area."""
    result: dict = {"components": [], "pins": [], "rows": [], "die_area": {}}
    if not def_path.exists():
        return result

    try:
        text = def_path.read_text(encoding="utf-8")
    except OSError:
        return result

    m = re.search(
        r"DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)", text,
    )
    if m:
        result["die_area"] = {
            "x1": int(m.group(1)),
            "y1": int(m.group(2)),
            "x2": int(m.group(3)),
            "y2": int(m.group(4)),
        }

    comp_section = re.search(
        r"COMPONENTS\s+\d+\s*;(.*?)END COMPONENTS", text, re.DOTALL,
    )
    if comp_section:
        for cm in re.finditer(
            r"-\s+(\S+)\s+(\S+)\s+.*?"
            r"(?:PLACED|FIXED)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+)",
            comp_section.group(1),
        ):
            result["components"].append({
                "name": cm.group(1),
                "cell_type": cm.group(2),
                "x": int(cm.group(3)),
                "y": int(cm.group(4)),
                "orient": cm.group(5),
            })

    pin_section = re.search(
        r"PINS\s+\d+\s*;(.*?)END PINS", text, re.DOTALL,
    )
    if pin_section:
        for pm in re.finditer(
            r"-\s+(\S+)\s+.*?DIRECTION\s+(\S+).*?"
            r"PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+)",
            pin_section.group(1),
        ):
            result["pins"].append({
                "name": pm.group(1),
                "direction": pm.group(2),
                "x": int(pm.group(3)),
                "y": int(pm.group(4)),
                "orient": pm.group(5),
            })

    for rm in re.finditer(
        r"ROW\s+(\S+)\s+\S+\s+(\d+)\s+(\d+)\s+\S+\s+"
        r"DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+)",
        text,
    ):
        result["rows"].append({
            "name": rm.group(1),
            "x": int(rm.group(2)),
            "y": int(rm.group(3)),
            "num_x": int(rm.group(4)),
            "num_y": int(rm.group(5)),
            "step_x": int(rm.group(6)),
            "step_y": int(rm.group(7)),
        })

    return result


# ---------------------------------------------------------------------------
# Synthesis report parser
# ---------------------------------------------------------------------------


def _parse_synthesis_report(report_path: Path) -> dict:
    """Parse cell-type distribution from a Yosys synthesis report.

    Handles multiple report formats:
    - LLM-generated: ``Total cells: N`` or ``Total Cells: N``
    - Legacy Yosys: ``Number of cells: N`` followed by ``<type> <count>``
    - Standard Yosys: ``Printing statistics.`` with ``<count> <area> cells``
    - ABC output: ``ABC RESULTS: <type> cells: <count>``
    """
    result: dict = {"cells": [], "total_cells": 0}
    if not report_path.exists():
        return result

    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError:
        return result

    # --- Strategy 0: LLM-generated "Total cells:" / "Total Cells:" ---
    tc_match = re.search(r"Total [Cc]ells:\s+(\d+)", text)
    if tc_match:
        result["total_cells"] = int(tc_match.group(1))

    # --- Strategy 1: legacy "Number of cells:" format ---
    m = re.search(r"Number of cells:\s+(\d+)", text)
    if m:
        result["total_cells"] = int(m.group(1))

    in_cells = False
    for line in text.split("\n"):
        if "Number of cells:" in line:
            in_cells = True
            continue
        if in_cells:
            cm = re.match(r"\s+(\S+)\s+(\d+)", line)
            if cm:
                result["cells"].append({
                    "type": cm.group(1),
                    "count": int(cm.group(2)),
                })
            elif line.strip() == "" or (
                line.strip() and not line.startswith(" ")
            ):
                if result["cells"]:
                    break

    if result["cells"]:
        return result

    # --- Strategy 2: "Printing statistics." section ---
    # Total line: ``  60  482.963 cells``  or  ``  16718 1.58E+05 cells``
    # Per-type:   ``   6   30.029   sky130_fd_sc_hd__a21oi_1``
    # Yosys prints submodule stats first, top-level last. We want the
    # LAST ``N <area> cells`` line which is the top-level total.
    _CELLS_RE = re.compile(
        r"^\s+(\d+)\s+[\d.eE+\-]+\s+cells\s*$", re.MULTILINE,
    )
    stats_match = re.search(r"Printing statistics\.", text)
    if stats_match:
        stats_text = text[stats_match.end():]
        all_cell_matches = list(_CELLS_RE.finditer(stats_text))
        tm = all_cell_matches[-1] if all_cell_matches else None
        if tm:
            result["total_cells"] = int(tm.group(1))
            cell_start = stats_text[tm.end():]
            for line in cell_start.split("\n"):
                cm = re.match(r"\s+(\d+)\s+[\d.eE+\-]+\s+(\S+)", line)
                if cm:
                    result["cells"].append({
                        "type": cm.group(2),
                        "count": int(cm.group(1)),
                    })
                elif line.strip() and not line.startswith(" "):
                    break

    if result["cells"]:
        return result

    # --- Strategy 3: "ABC RESULTS:" lines ---
    for am in re.finditer(r"ABC RESULTS:\s+(\S+)\s+cells:\s+(\d+)", text):
        result["cells"].append({
            "type": am.group(1),
            "count": int(am.group(2)),
        })
    if result["cells"]:
        result["total_cells"] = sum(c["count"] for c in result["cells"])

    return result


# ---------------------------------------------------------------------------
# Timing report parser
# ---------------------------------------------------------------------------


def _parse_timing_report(timing_path: Path) -> dict:
    """Parse critical path from OpenROAD ``timing_setup.rpt``."""
    result: dict = {
        "cells": [],
        "total_delay_ns": 0,
        "startpoint": "",
        "endpoint": "",
        "slack_ns": 0,
    }
    if not timing_path.exists():
        return result

    try:
        text = timing_path.read_text(encoding="utf-8")
    except OSError:
        return result

    m = re.search(r"Startpoint:\s+(\S+)", text)
    if m:
        result["startpoint"] = m.group(1)
    m = re.search(r"Endpoint:\s+(\S+)", text)
    if m:
        result["endpoint"] = m.group(1)

    m = re.search(r"(-?[\d.]+)\s+slack\s+\((?:MET|VIOLATED)\)", text)
    if m:
        result["slack_ns"] = float(m.group(1))

    for cm in re.finditer(
        r"\s+([\d.]+)\s+([\d.]+)\s+[v^]\s+(\S+)\s+\((\S+)\)", text,
    ):
        result["cells"].append({
            "delay": float(cm.group(1)),
            "arrival_ns": float(cm.group(2)),
            "pin": cm.group(3),
            "type": cm.group(4),
        })

    if result["cells"]:
        result["total_delay_ns"] = result["cells"][-1]["arrival_ns"]

    return result


# ---------------------------------------------------------------------------
# Power report parser
# ---------------------------------------------------------------------------


def _parse_power_report(power_path: Path) -> dict:
    """Parse power breakdown from OpenROAD ``power.rpt``."""
    result: dict = {
        "total_mw": 0,
        "internal_mw": 0,
        "switching_mw": 0,
        "leakage_mw": 0,
        "groups": [],
    }
    if not power_path.exists():
        return result

    try:
        text = power_path.read_text(encoding="utf-8")
    except OSError:
        return result

    for gm in re.finditer(
        r"(Sequential|Combinational|Macro|Pad)\s+"
        r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
        text,
    ):
        total_w = float(gm.group(5))
        if total_w > 0:
            result["groups"].append({
                "name": gm.group(1),
                "internal_mw": float(gm.group(2)) * 1000,
                "switching_mw": float(gm.group(3)) * 1000,
                "leakage_mw": float(gm.group(4)) * 1000,
                "total_mw": total_w * 1000,
            })

    tm = re.search(
        r"Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
        r"([\d.eE+-]+)\s+([\d.eE+-]+)",
        text,
    )
    if tm:
        result["internal_mw"] = float(tm.group(1)) * 1000
        result["switching_mw"] = float(tm.group(2)) * 1000
        result["leakage_mw"] = float(tm.group(3)) * 1000
        result["total_mw"] = float(tm.group(4)) * 1000

    return result


# ---------------------------------------------------------------------------
# Backend metrics aggregator
# ---------------------------------------------------------------------------


def _build_metrics(block_result: dict | None, pnr_dir: Path) -> dict:
    """Build a comprehensive metrics dict from block result and reports."""
    metrics: dict = {
        "total_cells": 0,
        "die_area_um2": 0,
        "design_area_um2": 0,
        "utilization_pct": 0,
        "wns_ns": 0,
        "tns_ns": 0,
        "setup_slack_ns": 0,
        "hold_slack_ns": 0,
        "timing_met": False,
        "total_power_mw": 0,
        "dynamic_power_mw": 0,
        "leakage_power_mw": 0,
        "drc_clean": False,
        "drc_violations": 0,
        "lvs_match": False,
        "lvs_device_delta": 0,
        "lvs_net_delta": 0,
    }

    if block_result:
        metrics.update({
            "die_area_um2": block_result.get("die_area_um2", 0),
            "design_area_um2": block_result.get("design_area_um2", 0),
            "utilization_pct": block_result.get("utilization_pct", 0),
            "wns_ns": block_result.get(
                "timing_wns_ns", block_result.get("wns_ns", 0),
            ),
            "tns_ns": block_result.get(
                "timing_tns_ns", block_result.get("tns_ns", 0),
            ),
            "setup_slack_ns": block_result.get("setup_slack_ns", 0),
            "hold_slack_ns": block_result.get("hold_slack_ns", 0),
            "timing_met": block_result.get("timing_met", False),
            "total_power_mw": block_result.get("total_power_mw", 0),
            "dynamic_power_mw": block_result.get("dynamic_power_mw", 0),
            "leakage_power_mw": block_result.get("leakage_power_mw", 0),
            "drc_clean": block_result.get("drc_clean", False),
            "drc_violations": block_result.get("drc_violations", 0),
            "lvs_match": block_result.get("lvs_match", False),
        })

    if pnr_dir.is_dir():
        from orchestrator.langgraph.backend_helpers import parse_openroad_reports

        pnr_metrics = parse_openroad_reports(str(pnr_dir))
        for k in (
            "design_area_um2", "die_area_um2", "utilization_pct",
            "wns_ns", "tns_ns", "setup_slack_ns", "hold_slack_ns",
            "total_power_mw", "dynamic_power_mw", "leakage_power_mw",
        ):
            if pnr_metrics.get(k, 0) and not metrics.get(k, 0):
                metrics[k] = pnr_metrics[k]
        if "timing_met" in pnr_metrics:
            metrics["timing_met"] = pnr_metrics["timing_met"]

    return metrics


# ---------------------------------------------------------------------------
# Pipeline timeline builder
# ---------------------------------------------------------------------------


def _build_timeline(events_path: Path) -> list[dict]:
    """Aggregate pipeline events into phase/step durations."""
    if not events_path.exists():
        return []

    events: list[dict] = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    phases: dict[str, dict[str, Any]] = {}
    node_starts: dict[str, float] = {}

    for evt in events:
        graph = evt.get("graph", "pipeline")
        node = evt.get("node", "")
        ts: float = evt.get("ts", 0)
        event_type = evt.get("event", "")

        if graph not in phases:
            phases[graph] = {"start": ts, "end": ts, "steps": {}}
        phases[graph]["end"] = max(phases[graph]["end"], ts)
        phases[graph]["start"] = min(phases[graph]["start"], ts)

        key = f"{graph}:{node}"
        if event_type == "graph_node_enter":
            node_starts[key] = ts
        elif event_type == "graph_node_exit" and key in node_starts:
            duration = ts - node_starts[key]
            phases[graph]["steps"][node] = {
                "name": node,
                "duration_s": round(duration, 2),
            }

    timeline: list[dict] = []
    for graph in sorted(phases):
        data = phases[graph]
        timeline.append({
            "phase": graph,
            "start_ts": data["start"],
            "end_ts": data["end"],
            "duration_s": round(data["end"] - data["start"], 2),
            "steps": list(data["steps"].values()),
        })
    return timeline


# ---------------------------------------------------------------------------
# Test results reader
# ---------------------------------------------------------------------------


def _read_test_results(root: Path, block_name: str) -> list[dict]:
    """Collect cocotb test pass/fail results for *block_name*.

    Returns dicts with ``passed`` (bool) and ``block`` keys so the
    Jinja2 template can use ``selectattr('passed')``.
    """
    results: list[dict] = []

    results_xml = root / "sim_build" / "results.xml"
    if results_xml.exists():
        try:
            text = results_xml.read_text(encoding="utf-8")
            for tm in re.finditer(
                r'<testcase\s+[^>]*name="(\w+)"[^>]*'
                r"(/>|>.*?</testcase>)",
                text,
                re.DOTALL,
            ):
                name = tm.group(1)
                body = tm.group(0)
                failed = "failure" in body.lower() or "error" in body.lower()
                results.append({
                    "name": name,
                    "passed": not failed,
                    "block": block_name,
                })
        except OSError:
            pass

    if not results:
        from orchestrator.langgraph.pipeline_helpers import _LOG_DIR
        log_dir = _LOG_DIR / block_name
        if log_dir.is_dir():
            for log_file in sorted(log_dir.glob("simulate_*.log")):
                try:
                    text = log_file.read_text(encoding="utf-8")
                    for tm in re.finditer(
                        r"(test_\w+)\s+(?:passed|PASSED)", text,
                    ):
                        results.append({
                            "name": tm.group(1),
                            "passed": True,
                            "block": block_name,
                        })
                    for tm in re.finditer(
                        r"(test_\w+)\s+(?:failed|FAILED)", text,
                    ):
                        results.append({
                            "name": tm.group(1),
                            "passed": False,
                            "block": block_name,
                        })
                except OSError:
                    continue

    return results


# ---------------------------------------------------------------------------
# Design-type inference for Example Output section
# ---------------------------------------------------------------------------


_TYPE_KEYWORDS: list[tuple[str, list[str], str]] = [
    (
        "arithmetic",
        ["adder", "multiplier", "alu", "arithmetic", "divider"],
        "Generate a Chart.js line chart showing the functional behavior of "
        "the arithmetic unit. For an adder: x-axis is 'Input A' (0 to 15), "
        "plot multiple lines for different values of Input B (0, 5, 10, 15), "
        "y-axis is 'Sum Output'. For a multiplier: use product. "
        "Title: 'Functional Behavior: Input vs Output'. "
        "Include a grid, legend, and clear axis labels. "
        "Use a light card background.",
    ),
    (
        "signal_processing",
        ["fft", "dft", "ifft", "fir", "iir", "filter", "dsp"],
        "Generate two stacked Chart.js line charts: top is time-domain input "
        "(sum of two sinusoids at 5 kHz and 12 kHz, 128 samples), bottom is "
        "frequency-domain output (magnitude spectrum with two peaks). "
        "Title: 'Signal Processing: Time Domain to Frequency Domain'.",
    ),
    (
        "codec",
        [
            "codec", "h264", "h265", "hevc", "avc", "jpeg",
            "encoder", "decoder", "video", "image",
        ],
        "Create a before/after visualization with two side-by-side 8x8 pixel "
        "grids using CSS grid cells. Left grid: 'Input Block' with a smooth "
        "gradient pattern of colors. Right grid: 'DCT Coefficients' with "
        "energy compaction (large top-left, near-zero elsewhere). Use a "
        "heatmap color scale (blue-white-red). "
        "Title: 'Codec Transform: Input Block to DCT Coefficients'.",
    ),
    (
        "processor",
        ["mcu", "cpu", "risc", "processor", "core", "rv32"],
        "Generate a Chart.js horizontal bar chart showing processor "
        "efficiency metrics: DMIPS/MHz, CoreMark/MHz, Area Efficiency "
        "(DMIPS/mm2), Power Efficiency (DMIPS/mW). Use representative "
        "values for the target technology. "
        "Title: 'Processor Performance Metrics'.",
    ),
    (
        "peripheral",
        ["uart", "spi", "i2c", "gpio", "pwm", "timer"],
        "Generate a Chart.js chart showing protocol timing waveforms "
        "using stepped line charts: clock, data, and control signals "
        "over ~20 clock cycles. Title: 'Protocol Timing Diagram'.",
    ),
]


def _infer_design_type(name: str, prd_text: str) -> tuple[str, str]:
    """Return ``(design_type, example_output_instructions)``."""
    combined = (name + " " + prd_text).lower()

    for dtype, keywords, hint in _TYPE_KEYWORDS:
        if any(kw in combined for kw in keywords):
            return dtype, hint

    return "generic", (
        "Generate a Chart.js radar chart showing the design's quality metrics "
        "normalized to 0-100: Timing Margin, Area Utilization, Power "
        "Efficiency, DRC Cleanliness, LVS Match, Test Coverage. "
        "Title: 'Design Quality Scorecard'."
    )


# ---------------------------------------------------------------------------
# VCD parser → WaveDrom JSON converter
# ---------------------------------------------------------------------------

_MAX_VCD_BYTES = 50 * 1024 * 1024  # skip files > 50 MB
_MAX_CLOCK_CYCLES = 512
_MAX_SIGNALS = 16


def _parse_vcd_header(lines: list[str]) -> tuple[dict[str, dict], int]:
    """Parse VCD header, returning (id_to_var_map, body_start_line_index).

    Each entry: identifier_code → {name, width, scope}
    """
    variables: dict[str, dict] = {}
    scope_stack: list[str] = []
    idx = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "$enddefinitions $end" or stripped == "$enddefinitions":
            idx += 1
            break
        if stripped.startswith("$scope"):
            parts = stripped.split()
            if len(parts) >= 3:
                scope_stack.append(parts[2])
        elif stripped.startswith("$upscope"):
            if scope_stack:
                scope_stack.pop()
        elif stripped.startswith("$var"):
            parts = stripped.replace("$end", "").split()
            if len(parts) >= 5:
                var_type = parts[1]
                width = int(parts[2])
                ident = parts[3]
                name = parts[4]
                if var_type in ("wire", "reg", "integer", "logic"):
                    variables[ident] = {
                        "name": name,
                        "width": width,
                        "scope": ".".join(scope_stack),
                    }
    return variables, idx


def _parse_vcd_values(
    lines: list[str],
    start_idx: int,
    variables: dict[str, dict],
) -> dict[str, list[tuple[int, str]]]:
    """Parse VCD value-change section, returning id → [(time, value), ...]."""
    changes: dict[str, list[tuple[int, str]]] = {k: [] for k in variables}
    current_time = 0

    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        if stripped.startswith("#"):
            try:
                current_time = int(stripped[1:])
            except ValueError:
                continue
        elif stripped.startswith(("b", "B")):
            parts = stripped.split()
            if len(parts) == 2:
                val = parts[0][1:]
                ident = parts[1]
                if ident in changes:
                    changes[ident].append((current_time, val))
        elif len(stripped) >= 2 and stripped[0] in "01xXzZ":
            val = stripped[0]
            ident = stripped[1:]
            if ident in changes:
                changes[ident].append((current_time, val))

    return changes


def _detect_clock_period(changes: dict[str, list[tuple[int, str]]], variables: dict[str, dict]) -> tuple[str, int]:
    """Find the clock signal (1-bit with shortest toggle period)."""
    best_id = ""
    best_period = float("inf")

    for ident, var in variables.items():
        if var["width"] != 1:
            continue
        transitions = changes.get(ident, [])
        if len(transitions) < 4:
            continue
        rising = [t for t, v in transitions if v == "1"]
        if len(rising) >= 2:
            period = rising[1] - rising[0]
            if 0 < period < best_period:
                best_period = period
                best_id = ident

    return best_id, int(best_period) if best_period != float("inf") else 0


def _vcd_to_wavedrom(vcd_path: Path) -> dict | None:
    """Convert a VCD file to WaveDrom JSON.

    Returns a WaveDrom-compatible dict or None if parsing fails.
    """
    if not vcd_path.exists():
        return None
    if vcd_path.stat().st_size > _MAX_VCD_BYTES:
        return None

    try:
        text = vcd_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.splitlines()
    variables, body_start = _parse_vcd_header(lines)
    if not variables:
        return None

    changes = _parse_vcd_values(lines, body_start, variables)
    clk_id, clk_period = _detect_clock_period(changes, variables)

    if not clk_period:
        return None

    all_times = set()
    for trans_list in changes.values():
        for t, _ in trans_list:
            all_times.add(t)
    vcd_end_time = max(all_times) if all_times else 0
    actual_cycles = min(vcd_end_time // clk_period + 1, _MAX_CLOCK_CYCLES)
    if actual_cycles < 2:
        actual_cycles = _MAX_CLOCK_CYCLES

    sample_times = [
        i * clk_period for i in range(actual_cycles + 1)
    ]
    sample_times[-1]

    toplevel_scope = ""
    scope_counts: dict[str, int] = {}
    for var in variables.values():
        s = var["scope"]
        scope_counts[s] = scope_counts.get(s, 0) + 1
    if scope_counts:
        toplevel_scope = max(scope_counts, key=scope_counts.get)

    sorted_vars = sorted(
        variables.items(),
        key=lambda kv: (kv[1]["scope"] != toplevel_scope, kv[1]["name"]),
    )

    signal_ids = []
    for ident, var in sorted_vars:
        if ident == clk_id:
            continue
        name_lower = var["name"].lower()
        if name_lower in ("clk", "clock"):
            continue
        if var["scope"] == toplevel_scope or not toplevel_scope:
            signal_ids.append(ident)
        if len(signal_ids) >= _MAX_SIGNALS - 1:
            break

    signals: list[dict] = []

    clk_wave = "p" + "." * (actual_cycles - 1)
    clk_name = variables[clk_id]["name"] if clk_id else "clk"
    signals.append({"name": clk_name, "wave": clk_wave})

    for ident in signal_ids:
        var = variables[ident]
        trans = changes.get(ident, [])
        if not trans:
            continue

        if var["width"] == 1:
            wave_chars = []
            current_val = "x"
            for t in sample_times[:-1]:
                for tt, vv in trans:
                    if tt <= t:
                        current_val = vv
                    else:
                        break
                if current_val in ("x", "X"):
                    wave_chars.append("x")
                elif current_val in ("z", "Z"):
                    wave_chars.append("z")
                elif current_val == "1":
                    wave_chars.append("1")
                else:
                    wave_chars.append("0")

            compressed = []
            last_val = None
            for ch in wave_chars:
                if ch == last_val:
                    compressed.append(".")
                else:
                    compressed.append(ch)
                    last_val = ch

            signals.append({"name": var["name"], "wave": "".join(compressed)})
        else:
            wave_chars = []
            data_vals: list[str] = []
            prev_val = None
            color_idx = 0
            colors = "2345"

            for t in sample_times[:-1]:
                current_val = "x"
                for tt, vv in trans:
                    if tt <= t:
                        current_val = vv
                    else:
                        break

                if current_val in ("x", "X"):
                    wave_chars.append("x")
                    prev_val = None
                elif current_val == prev_val:
                    wave_chars.append(".")
                else:
                    c = colors[color_idx % len(colors)]
                    wave_chars.append("=" if color_idx % 2 == 0 else c)
                    try:
                        int_val = int(current_val, 2)
                        data_vals.append(f"0x{int_val:X}")
                    except (ValueError, TypeError):
                        data_vals.append(current_val[:8])
                    prev_val = current_val
                    color_idx += 1

            entry: dict = {"name": var["name"], "wave": "".join(wave_chars)}
            if data_vals:
                entry["data"] = data_vals
            signals.append(entry)

    return {"signal": signals, "config": {"hscale": 2}, "foot": {"tick": 0}}


def _collect_vcd_waveforms(
    root: Path,
    completed_blocks: list[dict],
) -> list[dict]:
    """Collect and parse VCD files for all simulated blocks.

    Returns a list of {block_name, source, passed, wavedrom} dicts.
    """
    results: list[dict] = []

    for block in completed_blocks:
        name = block.get("name", "")
        if not name:
            continue
        vcd_path = root / "sim_build" / name / "dump.vcd"
        wavedrom = _vcd_to_wavedrom(vcd_path)
        if wavedrom:
            results.append({
                "block_name": name,
                "source": "block",
                "passed": block.get("sim_passed", block.get("success", False)),
                "wavedrom": wavedrom,
            })

    integration_vcd = root / "sim_build" / "integration" / "dump.vcd"
    wavedrom = _vcd_to_wavedrom(integration_vcd)
    if wavedrom:
        results.append({
            "block_name": "integration",
            "source": "integration",
            "passed": True,
            "wavedrom": wavedrom,
        })

    return results


# ---------------------------------------------------------------------------
# Deterministic VCD waveform HTML injection
# ---------------------------------------------------------------------------

_WAVEDROM_CDN = (
    '<script src="https://wavedrom.com/skins/default.js"></script>\n'
    '<script src="https://wavedrom.com/wavedrom.min.js"></script>'
)

_WAVEDROM_LAZY_JS = """\
<script>
(function() {
  function renderCard(btn) {
    var card = btn.closest('[data-vcd-card]');
    var body = card.querySelector('[data-vcd-body]');
    var arrow = btn.querySelector('[data-arrow]');
    if (body.style.display === 'none') {
      body.style.display = 'block';
      arrow.textContent = '▼';
      // Render WaveDrom for scripts not yet processed in this card
      var scripts = body.querySelectorAll('script[type="WaveDrom"]');
      scripts.forEach(function(s) {
        if (!s.dataset.rendered) {
          s.dataset.rendered = '1';
          // WaveDrom.RenderWaveForm replaces the script's parent container
          // so we wrap each in a fresh div and call ProcessAll scoped to it
          var wrapper = s.parentElement;
          WaveDrom.ProcessAll();
        }
      });
    } else {
      body.style.display = 'none';
      arrow.textContent = '▶';
    }
  }
  window._vcdToggle = renderCard;

  // Auto-expand the first card on load
  document.addEventListener('DOMContentLoaded', function() {
    var first = document.querySelector('[data-vcd-card] [data-vcd-toggle]');
    if (first) renderCard(first);
  });
})();
</script>
"""


def inject_vcd_waveforms(html: str, vcd_waveforms: list[dict]) -> str:
    """Inject WaveDrom VCD waveform section into dashboard HTML.

    Deterministic -- no LLM involved. Cards are collapsed by default
    with lazy WaveDrom rendering on expand to handle many blocks.
    The first card auto-expands on page load.
    """
    if not vcd_waveforms:
        return html

    if "wavedrom.min.js" not in html:
        html = html.replace("</head>", f"{_WAVEDROM_CDN}\n</head>")

    cards_html = ""
    for entry in vcd_waveforms:
        name = entry.get("block_name", "unknown")
        passed = entry.get("passed", False)
        wavedrom = entry.get("wavedrom", {})
        n_signals = len(wavedrom.get("signal", []))
        n_cycles = len(wavedrom.get("signal", [{}])[0].get("wave", ""))

        badge_cls = "badge-pass" if passed else "badge-fail"
        badge_txt = "PASS" if passed else "FAIL"
        source = entry.get("source", "block")
        source_label = (
            f"sim_build/{name}/dump.vcd"
            if source != "integration"
            else "sim_build/integration/dump.vcd"
        )
        wavedrom_json = json.dumps(wavedrom)

        cards_html += f"""
<div data-vcd-card style="background:var(--card,#1a1d2e);border:1px solid var(--border,#2a2f45);border-radius:8px;margin-bottom:.75rem">
<div data-vcd-toggle onclick="_vcdToggle(this)" style="display:flex;align-items:center;gap:.75rem;padding:1rem 1.25rem;cursor:pointer;user-select:none">
<span data-arrow style="font-size:.7rem;color:var(--dim,#8892b0);width:1rem">▶</span>
<span style="font-family:'JetBrains Mono',monospace;font-size:.95rem;font-weight:600">{name}</span>
<span class="{badge_cls}" style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:.72rem;font-weight:600">{badge_txt}</span>
<span style="margin-left:auto;font-size:.75rem;color:var(--dim,#8892b0)">{n_signals} signals &middot; {n_cycles} cycles</span>
</div>
<div data-vcd-body style="display:none;padding:0 1.25rem 1.25rem">
<div style="display:flex;gap:1.5rem;margin-bottom:.75rem;flex-wrap:wrap;font-size:.78rem;color:var(--dim,#8892b0)">
<span><strong style="color:var(--text,#e2e8f0)">Source:</strong> {source_label}</span>
</div>
<div style="overflow-x:auto;max-width:100%;padding-bottom:.5rem">
<script type="WaveDrom">
{wavedrom_json}
</script>
</div>
</div>
</div>
"""

    section_html = f"""
<!-- SIMULATION WAVEFORMS (VCD) — injected by coresmith pipeline -->
<section id="waveform">
<h2>Simulation Waveforms</h2>
<p style="color:var(--dim,#8892b0);font-size:.85rem;margin-bottom:1rem">
Actual signal traces from cocotb + Verilator simulation. Click a block to expand its waveform.
</p>
{cards_html}
</section>
{_WAVEDROM_LAZY_JS}
"""

    old_section = re.search(
        r'(?:<!-- (?:WAVEFORM|SIMULATION WAVEFORMS)[^>]*-->\s*)?'
        r'<section id="waveform">.*?</section>',
        html, re.DOTALL,
    )
    if old_section:
        html = html[:old_section.start()] + section_html + html[old_section.end():]
    elif "</main>" in html:
        html = html.replace("</main>", section_html + "\n</main>")
    elif "</body>" in html:
        html = html.replace("</body>", section_html + "\n</body>")

    html = re.sub(
        r"<script>\s*document\.addEventListener\(['\"]DOMContentLoaded['\"]"
        r"[^<]*?WaveDrom\.ProcessAll\(\)[^<]*?</script>",
        "",
        html,
    )

    return html


# ---------------------------------------------------------------------------
# HTML extraction helpers
# ---------------------------------------------------------------------------


def _extract_html(content: str) -> str:
    """Extract HTML from a ```html code fence in the LLM response."""
    match = re.search(r"```html\s*\n(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = content.strip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
        return stripped

    return _fallback_html(
        "LLM response did not contain an HTML code fence.",
    )


def _fallback_html(error: str) -> str:
    """Minimal error page when dashboard generation fails."""
    return (
        "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
        "<title>Dashboard Error</title>"
        "<style>body{font-family:sans-serif;background:#1a1a2e;color:#e0e0e0;"
        "display:flex;align-items:center;justify-content:center;min-height:100vh;}"
        ".msg{max-width:600px;padding:2rem;border:1px solid #333;border-radius:8px;}"
        "h1{color:#ff6b6b;}</style></head>"
        f"<body><div class='msg'><h1>Chip Finish Dashboard Generation Failed</h1>"
        f"<p>{error}</p></div></body></html>"
    )
