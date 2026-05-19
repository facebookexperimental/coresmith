# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Constraint Checker -- validates cross-cutting architectural constraints.

Combines deterministic pre-LLM checks (shuttle fit, GPIO pad budget) with
LLM-based holistic review.  The deterministic checks run first and are
always accurate; the LLM checks catch higher-level architectural issues
that rule-based checks would miss.

Each violation dict has:
  - violation: human-readable description string
  - category: "structural" (requires human input) or "auto_fixable" (LLM can iterate)
  - check: which constraint check flagged it (e.g. "peripheral_count")
  - severity: "error" or "warning"
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parents[1] / "langchain" / "prompts" / "constraint_check.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


def _get_shuttle_limits() -> dict:
    """Load shuttle physical limits from config.yaml tapeout section."""
    from orchestrator.langgraph.pipeline_helpers import load_config

    cfg = load_config()
    tapeout = cfg.get("tapeout", {})
    die_w = tapeout.get("die_width_um", 3520.0)
    die_h = tapeout.get("die_height_um", 5188.0)
    margin = tapeout.get("core_margin_um", 100.0)
    io_pads = tapeout.get("io_pads", 44)
    reserved_pads = 2  # GPIO[0]=clk, GPIO[1]=rst

    user_w = die_w - 2 * margin
    user_h = die_h - 2 * margin
    return {
        "target": tapeout.get("target", "openframe"),
        "die_width_um": die_w,
        "die_height_um": die_h,
        "die_area_mm2": (die_w * die_h) / 1e6,
        "user_width_um": user_w,
        "user_height_um": user_h,
        "user_area_mm2": (user_w * user_h) / 1e6,
        "core_margin_um": margin,
        "total_io_pads": io_pads,
        "reserved_pads": reserved_pads,
        "usable_io_pads": io_pads - reserved_pads,
    }


def _count_block_io_pads(block_diagram: dict) -> tuple[int, list[dict]]:
    """Count total GPIO pads needed by all blocks in the block diagram.

    Cross-references the ``connections`` array so that inter-block ports
    (which become internal wires, not chip-boundary pads) are excluded.

    Returns (total_pads_needed, per_block_details).
    """
    blocks = block_diagram.get("blocks", [])
    connections = block_diagram.get("connections", [])

    # Build set of (block_name, port_name) pairs that are internal wires
    connected_ports: set[tuple[str, str]] = set()
    for conn in connections:
        from_block = conn.get("from", conn.get("from_block", ""))
        to_block = conn.get("to", conn.get("to_block", ""))
        from_port = conn.get("from_port", conn.get("interface", ""))
        to_port = conn.get("to_port", conn.get("interface", ""))
        if from_block and from_port:
            connected_ports.add((from_block, from_port))
        if to_block and to_port:
            connected_ports.add((to_block, to_port))

    total = 0
    details: list[dict] = []

    for block in blocks:
        name = block.get("name", "unknown")
        interfaces = block.get("interfaces", {})
        block_pads = 0

        for port_name, port_info in interfaces.items():
            if port_name in ("clk", "rst", "rst_n"):
                continue
            if (name, port_name) in connected_ports:
                continue
            if isinstance(port_info, dict):
                width = port_info.get("width", 1)
            elif isinstance(port_info, int):
                width = port_info
            else:
                width = 1
            block_pads += width

        total += block_pads
        details.append({"block": name, "pads_needed": block_pads})

    return total, details


def _check_shuttle_constraints(
    block_diagram: dict,
    ers_spec: dict | None,
) -> list[dict]:
    """Deterministic shuttle constraint checks (GPIO pad budget + area fit).

    These run before the LLM and produce reliable, precise violations.
    """
    violations: list[dict] = []
    shuttle = _get_shuttle_limits()

    # --- Check #11: GPIO pad budget ---
    total_pads, pad_details = _count_block_io_pads(block_diagram)
    usable = shuttle["usable_io_pads"]

    if total_pads > usable:
        overflow = total_pads - usable
        detail_str = ", ".join(
            f"{d['block']}={d['pads_needed']}" for d in pad_details if d["pads_needed"] > 0
        )
        violations.append({
            "violation": (
                f"GPIO pad budget exceeded: design needs {total_pads} I/O pads "
                f"but {shuttle['target'].upper()} shuttle provides only {usable} "
                f"usable pads ({shuttle['total_io_pads']} total minus "
                f"{shuttle['reserved_pads']} reserved for clk/rst). "
                f"Overflow: {overflow} pads. "
                f"Per-block: {detail_str}. "
                f"Consider pin-muxing, serialization, or reducing interface widths."
            ),
            "category": "structural",
            "check": "gpio_pad_budget",
            "severity": "error",
        })
    elif total_pads > usable * 0.9:
        violations.append({
            "violation": (
                f"GPIO pad budget tight: design uses {total_pads}/{usable} "
                f"available pads ({total_pads * 100 / usable:.0f}% utilization). "
                f"Less than 10% headroom for debug/test pins."
            ),
            "category": "auto_fixable",
            "check": "gpio_pad_budget",
            "severity": "warning",
        })

    # --- Check #12: Shuttle area fit ---
    blocks = block_diagram.get("blocks", [])
    total_gates = sum(b.get("estimated_gates", 0) for b in blocks)

    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    area_budget = ers.get("area_budget", {}) or {}
    prd_die_area = area_budget.get("max_die_area_mm2")

    if prd_die_area and prd_die_area > shuttle["user_area_mm2"]:
        violations.append({
            "violation": (
                f"PRD die area budget ({prd_die_area:.3f} mm²) exceeds "
                f"{shuttle['target'].upper()} shuttle user area "
                f"({shuttle['user_area_mm2']:.3f} mm²). "
                f"Die dimensions: {shuttle['user_width_um']:.0f} x "
                f"{shuttle['user_height_um']:.0f} um. "
                f"The design will not fit in the shuttle frame. "
                f"Reduce area budget or choose a larger shuttle."
            ),
            "category": "structural",
            "check": "shuttle_area_fit",
            "severity": "error",
        })

    if total_gates > 0:
        gate_density_per_mm2 = 200_000  # ~200K gates/mm² for Sky130 HD at 60% util
        estimated_area_mm2 = total_gates / gate_density_per_mm2
        if estimated_area_mm2 > shuttle["user_area_mm2"]:
            violations.append({
                "violation": (
                    f"Estimated design area ({estimated_area_mm2:.3f} mm² from "
                    f"{total_gates:,} gates at ~200K gates/mm²) exceeds "
                    f"{shuttle['target'].upper()} shuttle user area "
                    f"({shuttle['user_area_mm2']:.3f} mm²). "
                    f"Reduce block count, gate estimates, or functionality."
                ),
                "category": "structural",
                "check": "shuttle_area_fit",
                "severity": "error",
            })
        elif estimated_area_mm2 > shuttle["user_area_mm2"] * 0.8:
            violations.append({
                "violation": (
                    f"Estimated design area ({estimated_area_mm2:.3f} mm²) uses "
                    f"{estimated_area_mm2 * 100 / shuttle['user_area_mm2']:.0f}% of "
                    f"shuttle user area ({shuttle['user_area_mm2']:.3f} mm²). "
                    f"Limited room for routing, power grid, and fill. "
                    f"Consider reducing gate count."
                ),
                "category": "auto_fixable",
                "check": "shuttle_area_fit",
                "severity": "warning",
            })

    return violations


def _near_domain_text(text: str, start: int, end: int) -> str:
    lo = max(0, start - 120)
    hi = min(len(text), end + 120)
    return text[lo:hi]


def _local_claim_text(text: str, start: int, end: int) -> str:
    """Return the local clause/sentence that owns a numeric claim."""
    delimiters = ".;\n\r{}[]"
    lo = start
    while lo > 0 and text[lo - 1] not in delimiters:
        lo -= 1
    hi = end
    while hi < len(text) and text[hi] not in delimiters:
        hi += 1
    return text[lo:hi].lower()


def _parse_claim_int(value: str) -> int:
    return int(value.replace(",", ""))


def _looks_like_physical_dimension(context: str) -> bool:
    return any(
        token in context
        for token in (
            "die dimension",
            "die area",
            "user area",
            "user_width",
            "user_height",
            "core_margin",
            "openframe",
            "shuttle",
            "sky130",
            " mm",
            "mm²",
            " um",
            "µm",
            "metal",
            "routing",
        )
    )


def _extract_json_numbers(payload: Any) -> dict[str, int]:
    """Extract numeric facts from structured architecture artifacts.

    This is intentionally schema-tolerant. Architecture agents use different
    names for the same concepts, so the deterministic constraint checker looks
    for common semantic keys instead of one codec-specific schema.
    """
    out: dict[str, int] = {}

    def visit(value: Any, key_path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{key_path}.{key}" if key_path else str(key)
                visit(item, child)
            return
        if isinstance(value, list):
            for idx, item in enumerate(value):
                visit(item, f"{key_path}[{idx}]")
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            key = key_path.lower()
            intval = int(value)
            if any(token in key for token in ("width", "frame_w", "image_w", "matrix_m")):
                out.setdefault("width", intval)
            if any(token in key for token in ("height", "frame_h", "image_h", "matrix_n")):
                out.setdefault("height", intval)
            if any(token in key for token in ("block_width", "tile_width", "macroblock_width")):
                out.setdefault("block_width", intval)
            if any(token in key for token in ("block_height", "tile_height", "macroblock_height")):
                out.setdefault("block_height", intval)
            if any(token in key for token in ("columns", "cols", "mb_cols", "block_cols")):
                out.setdefault("columns", intval)
            if any(token in key for token in ("rows", "mb_rows", "block_rows")):
                out.setdefault("rows", intval)
            if key.endswith("mb_x") or key.endswith("x_max") or "mb_x_max" in key:
                out.setdefault("mb_x_max", intval)
            if key.endswith("mb_y") or key.endswith("y_max") or "mb_y_max" in key:
                out.setdefault("mb_y_max", intval)

    visit(payload)
    return out


def _artifact_text(
    *,
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    requirements: str,
    ers_spec: dict | None,
    project_root: str,
) -> tuple[str, dict[str, int]]:
    parts = [requirements]
    structured = {
        "block_diagram": block_diagram,
        "memory_map": memory_map,
        "clock_tree": clock_tree,
        "register_spec": register_spec,
        "ers_spec": ers_spec or {},
    }
    facts = _extract_json_numbers(structured)
    parts.append(json.dumps(structured, indent=2, default=str))

    root = Path(project_root)
    read_project_docs = project_root not in ("", ".")
    if read_project_docs:
        for rel in (
            "arch/prd_spec.md",
            "arch/sad_spec.md",
            "arch/frd_spec.md",
            "arch/block_diagram.md",
            "arch/ers_spec.md",
        ):
            path = root / rel
            if path.exists():
                try:
                    parts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    pass
        uarch_dir = root / "arch" / "uarch_specs"
        if uarch_dir.exists():
            for path in sorted(uarch_dir.glob("*.md")):
                try:
                    parts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    pass

    return "\n".join(parts), facts


def _extract_dimension_facts(text: str, structured_facts: dict[str, int]) -> dict[str, Any]:
    lower = text.lower()
    facts: dict[str, Any] = {"structured": structured_facts}

    dims: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\b(\d{2,5})\s*[x×]\s*(\d{2,5})\b", text, flags=re.IGNORECASE):
        context = _near_domain_text(lower, match.start(), match.end())
        immediate = lower[max(0, match.start() - 40): min(len(lower), match.end() + 40)]
        if _looks_like_physical_dimension(immediate):
            continue
        if any(token in context for token in ("frame", "image", "video", "pixel", "matrix", "tensor", "sample")):
            w = int(match.group(1))
            h = int(match.group(2))
            dims.append((w, h, match.group(0)))
    if "width" in structured_facts and "height" in structured_facts:
        dims.append((structured_facts["width"], structured_facts["height"], "structured width/height"))
    facts["source_dims"] = dims
    if dims:
        # Prefer the largest plausible 2-D data object over small block sizes.
        facts["source_dim"] = max(dims, key=lambda item: item[0] * item[1])

    blocks: list[tuple[int, int, str]] = []
    for match in re.finditer(
        r"\b(\d{1,4})\s*[x×]\s*(\d{1,4})\s+(?:macroblocks?|blocks?|tiles?|subblocks?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        bw = int(match.group(1))
        bh = int(match.group(2))
        if bw > 0 and bh > 0:
            blocks.append((bw, bh, match.group(0)))
    for match in re.finditer(
        r"\b(?:macroblock|block|tile)[-_ ]?(?:size|dimensions?)\D{0,40}(\d{1,4})\s*[x×]\s*(\d{1,4})\b",
        text,
        flags=re.IGNORECASE,
    ):
        bw = int(match.group(1))
        bh = int(match.group(2))
        if bw > 0 and bh > 0:
            blocks.append((bw, bh, match.group(0)))
    if "block_width" in structured_facts and "block_height" in structured_facts:
        blocks.append((structured_facts["block_width"], structured_facts["block_height"], "structured block width/height"))
    facts["block_dims"] = blocks
    if blocks:
        # Prefer small square-ish block/tile sizes over frame dimensions.
        facts["block_dim"] = min(blocks, key=lambda item: item[0] * item[1])

    col_row_claims: list[dict[str, Any]] = []
    patterns = [
        r"\b(\d{1,6})\s+(?:macroblock\s+|block\s+|tile\s+)?columns?.{0,80}?\b(\d{1,6})\s+(?:macroblock\s+|block\s+|tile\s+)?rows?\b",
        r"\b(\d{1,6})\s+(?:macroblocks?|blocks?|tiles?)\s+per\s+row.{0,80}?\b(\d{1,6})\s+(?:rows?|row groups?)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            context = _near_domain_text(lower, match.start(), match.end())
            if "pixel" in context and not any(token in context for token in ("macroblock", "block", "tile")):
                continue
            col_row_claims.append({
                "columns": int(match.group(1)),
                "rows": int(match.group(2)),
                "source": " ".join(match.group(0).split())[:180],
            })
    if "columns" in structured_facts and "rows" in structured_facts:
        col_row_claims.append({
            "columns": structured_facts["columns"],
            "rows": structured_facts["rows"],
            "source": "structured columns/rows",
        })
    facts["col_row_claims"] = col_row_claims

    coord_claims: list[dict[str, Any]] = []
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:x|_x|_col|col))\s*[=:]?\s*0\s*\.\.\s*(\d{1,6})",
        text,
    ):
        name = match.group(1)
        lname = name.lower()
        if any(token in lname for token in ("pixel", "coeff", "quant", "sample", "byte", "bit", "addr", "index")):
            continue
        if any(token in lname for token in ("subblock", "local", "inner", "intra")):
            continue
        if not any(token in lname for token in ("mb", "macroblock", "block", "tile", "col")):
            continue
        coord_claims.append({"axis": "x", "name": name, "max": int(match.group(2)), "source": match.group(0)})
    for match in re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*(?:y|_y|_row|row))\s*[=:]?\s*0\s*\.\.\s*(\d{1,6})",
        text,
    ):
        name = match.group(1)
        lname = name.lower()
        if any(token in lname for token in ("pixel", "coeff", "quant", "sample", "byte", "bit", "addr", "index")):
            continue
        if any(token in lname for token in ("subblock", "local", "inner", "intra")):
            continue
        if not any(token in lname for token in ("mb", "macroblock", "block", "tile", "row")):
            continue
        coord_claims.append({"axis": "y", "name": name, "max": int(match.group(2)), "source": match.group(0)})
    facts["coord_claims"] = coord_claims

    total_claims: list[dict[str, Any]] = []
    for match in re.finditer(
        r"\b(?:exactly\s+)?(\d{1,3}(?:,\d{3})+|\d{2,8})\s+(?:macroblocks?|blocks?|tiles?|transactions?)\s+(?:per\s+frame|total|in\s+total|for\b)",
        text,
        flags=re.IGNORECASE,
    ):
        context = _local_claim_text(lower, match.start(), match.end())
        prefix = lower[max(0, match.start() - 20):match.start()]
        if re.search(r"(?:\bby|[x×*])\s*$", prefix):
            continue
        if any(
            token in context
            for token in (
                "previous",
                "stale",
                "violation",
                "before/after",
                "was fixed",
                "removed",
                "obsolete",
                "incorrect",
                "wrong",
            )
        ):
            continue
        total_claims.append({"total": _parse_claim_int(match.group(1)), "source": " ".join(match.group(0).split())})
    facts["total_claims"] = total_claims

    width_claims: list[dict[str, Any]] = []
    for coord in ("mb_x", "mb_y", "x", "y", "col", "row"):
        for match in re.finditer(
            rf"\b{re.escape(coord)}\b.{{0,80}}?\b(\d{{1,2}})\s*bits?\b",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            width_claims.append({"name": coord, "bits": int(match.group(1)), "source": " ".join(match.group(0).split())[:160]})
    facts["width_claims"] = width_claims
    return facts


def _select_derived_geometry_pair(facts: dict[str, Any]) -> tuple[tuple[int, int, str], tuple[int, int, str]] | tuple[None, None]:
    """Select source/block dimensions that best explain explicit claims.

    Architecture artifacts often mention multiple 2-D domains: video frame
    size, physical die size, transform sub-blocks, SRAM shapes, and tile grids.
    The derived checker should validate the contract domain whose dimensions
    agree with the artifact's own columns/rows/ranges/totals instead of blindly
    pairing the largest source dimension with the smallest block dimension.
    """
    sources = list(facts.get("source_dims") or [])
    blocks = list(facts.get("block_dims") or [])
    if not sources and facts.get("source_dim"):
        sources = [facts["source_dim"]]
    if not blocks and facts.get("block_dim"):
        blocks = [facts["block_dim"]]

    best: tuple[int, tuple[int, int, str], tuple[int, int, str]] | None = None
    for source in sources:
        width, height, _dim_src = source
        for block in blocks:
            block_w, block_h, block_src = block
            if block_w <= 0 or block_h <= 0:
                continue
            if width % block_w != 0 or height % block_h != 0:
                continue

            expected_cols = width // block_w
            expected_rows = height // block_h
            expected_total = expected_cols * expected_rows
            score = 0

            for claim in facts.get("col_row_claims", []):
                if claim["columns"] == expected_cols and claim["rows"] == expected_rows:
                    score += 40
                else:
                    score -= 2

            for total in facts.get("total_claims", []):
                if total["total"] == expected_total:
                    score += 20
                else:
                    score -= 1

            for coord in facts.get("coord_claims", []):
                expected_max = expected_cols - 1 if coord["axis"] == "x" else expected_rows - 1
                if coord["max"] == expected_max:
                    score += 8
                else:
                    score -= 1

            block_text = block_src.lower()
            if "macroblock" in block_text:
                score += 8
            if "tile" in block_text:
                score += 5
            if "subblock" in block_text:
                score -= 8
            if "structured" in block_text:
                score += 4

            source_text = source[2].lower()
            if "structured" in source_text:
                score += 4

            if best is None or score > best[0]:
                best = (score, source, block)

    if best is None:
        source = facts.get("source_dim")
        block = facts.get("block_dim")
        return (source, block) if source and block else (None, None)

    return best[1], best[2]


def _check_derived_constraints(
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    requirements: str,
    ers_spec: dict | None,
    project_root: str,
) -> list[dict]:
    """Check generic derived arithmetic and contract consistency.

    This is deliberately domain-agnostic: it validates facts such as
    dimensions/tile counts/coordinate ranges/field widths once the artifacts
    themselves introduce those concepts. The codec bug was one instance of this
    class: width-derived columns and height-derived rows were transposed.
    """
    text, structured = _artifact_text(
        block_diagram=block_diagram,
        memory_map=memory_map,
        clock_tree=clock_tree,
        register_spec=register_spec,
        requirements=requirements,
        ers_spec=ers_spec,
        project_root=project_root,
    )
    facts = _extract_dimension_facts(text, structured)
    violations: list[dict] = []

    source_dim, block_dim = _select_derived_geometry_pair(facts)
    if source_dim and block_dim:
        width, height, dim_src = source_dim
        block_w, block_h, block_src = block_dim
        if block_w > 0 and block_h > 0 and width % block_w == 0 and height % block_h == 0:
            expected_cols = width // block_w
            expected_rows = height // block_h
            expected_total = expected_cols * expected_rows

            for claim in facts.get("col_row_claims", []):
                cols = claim["columns"]
                rows = claim["rows"]
                claim_source = str(claim.get("source", "")).lower()
                if (
                    cols == width
                    and rows == height
                    and not any(token in claim_source for token in ("macroblock", "block", "tile"))
                ):
                    continue
                if cols != expected_cols or rows != expected_rows:
                    violations.append({
                        "violation": (
                            "Derived geometry mismatch: source dimensions "
                            f"{width}x{height} from {dim_src!r} and block dimensions "
                            f"{block_w}x{block_h} from {block_src!r} require "
                            f"{expected_cols} columns and {expected_rows} rows, but artifact "
                            f"claims {cols} columns and {rows} rows ({claim['source']}). "
                            "Width-derived counts are columns/x; height-derived counts are rows/y."
                        ),
                        "category": "auto_fixable",
                        "check": "derived_geometry_columns_rows",
                        "severity": "error",
                    })

            for coord in facts.get("coord_claims", []):
                expected_max = expected_cols - 1 if coord["axis"] == "x" else expected_rows - 1
                if coord["max"] != expected_max:
                    violations.append({
                        "violation": (
                            f"Derived coordinate range mismatch: {coord['name']} range "
                            f"{coord['source']} implies max {coord['max']}, but "
                            f"{width}x{height} split into {block_w}x{block_h} blocks "
                            f"requires {coord['axis']} max {expected_max}."
                        ),
                        "category": "auto_fixable",
                        "check": "derived_coordinate_range",
                        "severity": "error",
                    })

            for total in facts.get("total_claims", []):
                if total["total"] != expected_total:
                    violations.append({
                        "violation": (
                            f"Derived transaction count mismatch: {width}x{height} split into "
                            f"{block_w}x{block_h} blocks requires {expected_total} total "
                            f"blocks/transactions, but artifact claims {total['total']} "
                            f"({total['source']})."
                        ),
                        "category": "auto_fixable",
                        "check": "derived_transaction_count",
                        "severity": "error",
                    })

            coord_max = {"mb_x": expected_cols - 1, "x": expected_cols - 1, "col": expected_cols - 1,
                         "mb_y": expected_rows - 1, "y": expected_rows - 1, "row": expected_rows - 1}
            for width_claim in facts.get("width_claims", []):
                name = width_claim["name"].lower()
                max_value = coord_max.get(name)
                if max_value is None:
                    continue
                needed = max(1, math.ceil(math.log2(max_value + 1)))
                if width_claim["bits"] < needed:
                    violations.append({
                        "violation": (
                            f"Coordinate field width too small: {width_claim['source']} gives "
                            f"{width_claim['bits']} bits, but max {max_value} requires at least "
                            f"{needed} bits."
                        ),
                        "category": "auto_fixable",
                        "check": "derived_coordinate_width",
                        "severity": "error",
                    })

    try:
        socmate_dir = Path(project_root) / ".socmate"
        socmate_dir.mkdir(parents=True, exist_ok=True)
        (socmate_dir / "derived_constraints_audit.json").write_text(
            json.dumps({"facts": facts, "violations": violations}, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError:
        pass

    return violations


def _extract_clock_mhz(text: str, ers_spec: dict | None) -> float | None:
    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    speed = ers.get("speed_and_feeds", {}) or {}
    for key in ("target_clock_mhz", "clock_mhz"):
        value = speed.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        if isinstance(value, str):
            match = re.search(r"(\d+(?:\.\d+)?)", value)
            if match:
                return float(match.group(1))

    match = re.search(r"\b(\d+(?:\.\d+)?)\s*MHz\b", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_fps(text: str, ers_spec: dict | None) -> float | None:
    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    candidates = "\n".join(_walk_text({
        "speed_and_feeds": ers.get("speed_and_feeds", {}),
        "kpis": ers.get("kpis", ers.get("validation_kpis", [])),
        "requirements": ers.get("requirements", []),
        "text": text,
    }))
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:fps|frames\s*/\s*s|frames\s+per\s+second)\b", candidates, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _check_performance_constraints(
    *,
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    requirements: str,
    ers_spec: dict | None,
    project_root: str,
) -> list[dict]:
    """Check that stated cycle budgets do not contradict frame-rate KPIs."""
    text, structured = _artifact_text(
        block_diagram=block_diagram,
        memory_map=memory_map,
        clock_tree=clock_tree,
        register_spec=register_spec,
        requirements=requirements,
        ers_spec=ers_spec,
        project_root=project_root,
    )
    facts = _extract_dimension_facts(text, structured)
    source_dim, block_dim = _select_derived_geometry_pair(facts)
    clock_mhz = _extract_clock_mhz(text, ers_spec)
    fps = _extract_fps(text, ers_spec)
    if not source_dim or not block_dim or not clock_mhz or not fps:
        return []

    width, height, _ = source_dim
    block_w, block_h, _ = block_dim
    if block_w <= 0 or block_h <= 0 or width % block_w != 0 or height % block_h != 0:
        return []

    cols = width // block_w
    rows = height // block_h
    total_transactions = cols * rows
    if total_transactions <= 0:
        return []

    cycles_per_frame_budget = int((clock_mhz * 1_000_000) // fps)
    cycles_per_transaction_budget = cycles_per_frame_budget / total_transactions
    violations: list[dict] = []

    cycle_claims: list[tuple[int, str]] = []
    for match in re.finditer(
        r"\b(\d{1,9})\s+cycles?\s+per\s+(?:macroblock|block|tile|transaction)\b",
        text,
        flags=re.IGNORECASE,
    ):
        cycle_claims.append((int(match.group(1)), " ".join(match.group(0).split())))

    for match in re.finditer(
        r"\bwithin\s+(\d{1,9})\s*\+\s*(\d{1,9})\s*\*\s*(?:\d{1,9}|macroblocks?|blocks?|tiles?|transactions?)\s+cycles?\b",
        text,
        flags=re.IGNORECASE,
    ):
        base = int(match.group(1))
        per = int(match.group(2))
        total = base + per * total_transactions
        if total > cycles_per_frame_budget:
            violations.append({
                "violation": (
                    f"Throughput KPI contradiction: {clock_mhz:g} MHz and {fps:g} fps allow "
                    f"at most {cycles_per_frame_budget} cycles/frame for {width}x{height}. "
                    f"The artifact claims {match.group(0)!r}, which implies {total} cycles/frame "
                    f"for {total_transactions} transactions. Repair the block diagram with an "
                    "explicit cycle budget that satisfies the KPI or ask a blocking question."
                ),
                "category": "auto_fixable",
                "check": "performance_cycle_budget",
                "severity": "error",
            })

    for cycles, source in cycle_claims:
        if cycles > cycles_per_transaction_budget:
            violations.append({
                "violation": (
                    f"Throughput KPI contradiction: {clock_mhz:g} MHz and {fps:g} fps allow "
                    f"at most {cycles_per_frame_budget} cycles/frame, or "
                    f"{cycles_per_transaction_budget:.1f} cycles per {block_w}x{block_h} transaction "
                    f"for {total_transactions} transactions/frame. The artifact claims {source}. "
                    "Repair the block diagram with explicit pipeline throughput arithmetic."
                ),
                "category": "auto_fixable",
                "check": "performance_cycle_budget",
                "severity": "error",
            })

    return violations


def _shuttle_constraints_enabled(requirements: str = "", ers_spec: dict | None = None) -> bool:
    """Return whether package/shuttle constraints should be enforced.

    Most SocMate frontend runs generate reusable soft IP, where block ports are
    internal module interfaces rather than package GPIO. Shuttle pad/area checks
    are useful for MPW benchmark wrappers, but they must be opt-in to avoid
    forcing every soft IP through an OpenFrame pin budget.
    """
    value = os.environ.get("SOCMATE_ENABLE_SHUTTLE_CONSTRAINTS", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    text = requirements.lower()
    if any(token in text for token in ("openframe wrapper", "mpw wrapper", "shuttle gpio")):
        return True

    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    target = json.dumps(ers.get("technology", {})).lower()
    return any(token in target for token in ("openframe", "caravel", "mpw"))


async def check_constraints(
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    benchmark_results: dict | None = None,
    pdk_config: dict | None = None,
    requirements: str = "",
    ers_spec: dict | None = None,
    project_root: str = ".",
) -> list[dict]:
    """Validate cross-cutting architectural constraints.

    Runs deterministic shuttle checks first (GPIO pad budget, area fit),
    then delegates holistic review to the LLM for rules 1-10.

    Args:
        block_diagram: Block diagram with blocks and connections.
        memory_map: Memory map with peripherals and address allocations.
        clock_tree: Clock tree with domains and crossings.
        register_spec: Register specifications per block.
        benchmark_results: Benchmark synthesis results for gate estimates.
        pdk_config: PDK configuration dict.
        requirements: High-level system requirements for context.
        ers_spec: Product Requirements Document (structured dict).

    Returns:
        List of violation dicts. Each dict has keys: violation, category,
        check, severity. Empty list = all constraints pass.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.constraints")

    with tracer.start_as_current_span("check_constraints") as span:
        # --- Deterministic shuttle checks (opt-in for package/top-level runs) ---
        shuttle_enabled = _shuttle_constraints_enabled(requirements, ers_spec)
        shuttle_violations = (
            _check_shuttle_constraints(block_diagram, ers_spec)
            if shuttle_enabled
            else []
        )
        derived_violations = _check_derived_constraints(
            block_diagram=block_diagram,
            memory_map=memory_map,
            clock_tree=clock_tree,
            register_spec=register_spec,
            requirements=requirements,
            ers_spec=ers_spec,
            project_root=project_root,
        )
        performance_violations = _check_performance_constraints(
            block_diagram=block_diagram,
            memory_map=memory_map,
            clock_tree=clock_tree,
            register_spec=register_spec,
            requirements=requirements,
            ers_spec=ers_spec,
            project_root=project_root,
        )

        # --- Extract PRD-driven limits ---
        ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
        area = ers.get("area_budget", {}) or {}
        dataflow = ers.get("dataflow", {}) or {}

        max_gate_count = _safe_int(area.get("max_gate_count"), 2_000_000)
        sram_size_kb = _extract_sram_budget_kb(ers, default=32)
        bus_protocol = dataflow.get("bus_protocol", "")
        block_count = len(block_diagram.get("blocks", []))
        max_peripheral_count = min(block_count + 2, 15) if block_count > 8 else 8
        memory_map_enabled = bool((memory_map.get("result", memory_map) or {}).get("peripherals"))
        clock_tree_enabled = bool((clock_tree.get("result", clock_tree) or {}).get("domains"))
        register_spec_enabled = bool(
            (register_spec.get("result", register_spec) or {}).get("register_blocks")
        )

        # Build gate budget rationale
        die_area = area.get("max_die_area_mm2")
        if die_area:
            gate_budget_rationale = (
                f"{die_area} mm² die at 60% utilization"
            )
        else:
            gate_budget_rationale = "default budget (no ERS die area specified)"

        shuttle = _get_shuttle_limits()
        total_pads, _ = _count_block_io_pads(block_diagram)
        if shuttle_enabled:
            shuttle_rules = (
                "SHUTTLE CONSTRAINT RULES:\n\n"
                f"11. GPIO PAD BUDGET: This design targets the {shuttle['target'].upper()} "
                f"shuttle with {shuttle['usable_io_pads']} usable I/O pads "
                f"(GPIO[0]=clk, GPIO[1]=rst reserved). Currently {total_pads} pads needed. "
                f"If any block's interface widths would cause GPIO overflow, flag as "
                f"\"structural\" severity \"error\".\n\n"
                f"12. SHUTTLE AREA FIT: The {shuttle['target'].upper()} shuttle user area is "
                f"{shuttle['user_area_mm2']:.3f} mm² ({shuttle['user_width_um']:.0f} x "
                f"{shuttle['user_height_um']:.0f} um). Sum of block estimated_gates / 200000 "
                f"must be < user_area * 0.8 for routing margin. Flag as \"structural\" "
                f"severity \"error\" if exceeded.\n"
            )
        else:
            shuttle_rules = (
                "SHUTTLE CONSTRAINT RULES:\n"
                "11-12. Skipped for this run. The architecture target is reusable "
                "soft IP, so block interfaces are internal RTL ports rather than "
                "package GPIO. Enforce shuttle GPIO/area only when "
                "SOCMATE_ENABLE_SHUTTLE_CONSTRAINTS=1 or the requirements "
                "explicitly target an MPW/OpenFrame/Caravel wrapper.\n"
            )

        # --- Build bus rules (skip for simple designs) ---
        needs_bus = (
            block_count >= 3
            and bus_protocol not in ("none", "None", "N/A", "")
        )
        if needs_bus:
            bus_rules = (
                "BUS AND MEMORY CONSTRAINT RULES:\n\n"
                f"1. PERIPHERAL COUNT: The peripheral bus uses nibble decode.\n"
                f"   Maximum {max_peripheral_count} peripherals. If exceeded, flag as\n"
                f"   \"structural\" severity \"error\".\n\n"
                "2. MEMORY OVERLAP: No two address regions may overlap.\n"
                "   Check every pair. If overlap found, flag as \"structural\" severity \"error\".\n\n"
                f"3. SRAM BUDGET: SRAM is {sram_size_kb}KB ({sram_size_kb * 1024} bytes).\n"
                f"   If total estimated usage exceeds {sram_size_kb}KB, flag as \"structural\" severity \"error\".\n\n"
                "6. BLOCK CONNECTIVITY: Every non-infrastructure block must appear in at least\n"
                "   one connection. Orphaned blocks are \"structural\" severity \"error\"."
            )
        else:
            bus_rules = (
                "BUS AND MEMORY CONSTRAINT RULES:\n"
                "(Skipped -- this design has fewer than 3 blocks or no bus protocol.)"
            )

        optional_stage_note = (
            "OPTIONAL STAGE NOTE:\n"
            "- Memory-map checks are disabled for this run unless a non-empty "
            "memory map is present.\n"
            "- Clock-domain and CDC checks are disabled for this run unless a "
            "non-empty clock tree is present.\n"
            "- Register-spec consistency checks are disabled for this run unless "
            "a non-empty register spec is present.\n\n"
        )

        additional_rules = (
            optional_stage_note +
            "ADDITIONAL CONSTRAINT RULES:\n\n"
            + (
                "4. CLOCK DOMAIN CROSSINGS: If the clock tree has multiple domains, every\n"
                "   connection between blocks in different domains MUST have a CDC module.\n"
                "   Missing CDC is \"auto_fixable\" severity \"error\".\n\n"
                if clock_tree_enabled
                else "4. CLOCK DOMAIN CROSSINGS: skipped because clock-tree generation is disabled.\n\n"
            ) +
            f"5. GATE BUDGET: Total estimated gates must not exceed {max_gate_count:,}\n"
            f"   ({gate_budget_rationale}). If exceeded, flag as \"auto_fixable\" severity \"error\".\n\n"
            "ADDITIONAL REVIEW (go beyond the rules above):\n"
            + (
                "7. Check that register spec blocks match memory map peripherals.\n"
                if register_spec_enabled and memory_map_enabled
                else "7. Register/memory-map consistency skipped because one or both optional artifacts are disabled.\n"
            ) +
            (
                "8. Check that all blocks in the block diagram appear in exactly one clock domain.\n"
                if clock_tree_enabled
                else "8. Per-block clock-domain membership skipped because clock-tree generation is disabled.\n"
            ) +
            "9. Check for data width mismatches in connections (source width != dest width\n"
            "   without an adapter block).\n"
            "10. Check that tier assignments are reasonable (e.g., an FFT should not be tier 1).\n\n"
            + shuttle_rules
        )

        parts = [
            "Review the following ASIC architecture for constraint violations.",
            "Check ALL rules listed in your instructions (1-12).",
        ]
        if requirements:
            parts.append(f"\nSystem requirements: {requirements}")

        parts.append(
            f"\n--- BLOCK DIAGRAM ---\n{json.dumps(block_diagram, indent=2)}"
        )
        parts.append(
            f"\n--- MEMORY MAP ---\n{json.dumps(memory_map, indent=2)}"
        )
        parts.append(
            f"\n--- CLOCK TREE ---\n{json.dumps(clock_tree, indent=2)}"
        )
        parts.append(
            f"\n--- REGISTER SPEC ---\n{json.dumps(register_spec, indent=2)}"
        )

        if benchmark_results:
            parts.append(
                f"\n--- BENCHMARK DATA ---\n{json.dumps(benchmark_results, indent=2)}"
            )
        if pdk_config:
            parts.append(
                f"\n--- PDK CONFIG ---\n{json.dumps(pdk_config, indent=2)}"
            )

        from pathlib import Path as _P
        _root = _P(project_root)
        for doc_name, doc_label in [
            ("sad_spec.md", "SYSTEM ARCHITECTURE DOCUMENT (SAD)"),
            ("frd_spec.md", "FUNCTIONAL REQUIREMENTS DOCUMENT (FRD)"),
        ]:
            doc_path = _root / "arch" / doc_name
            if doc_path.exists():
                try:
                    parts.append(f"\n--- {doc_label} ---\n{doc_path.read_text()}")
                except OSError:
                    pass

        user_message = "\n".join(parts)

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=600)

        # Fill template variables in system prompt
        system_prompt = SYSTEM_PROMPT.format(
            bus_rules=bus_rules,
            additional_rules=additional_rules,
        )

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="constraint_check",
            )
            result = _parse_response(content)
            llm_violations = result.get("violations", [])

            # Merge: deterministic violations first, then LLM violations.
            # Deduplicate: skip LLM violations for checks already covered
            deterministic_violations = shuttle_violations + derived_violations + performance_violations
            covered_checks = {v["check"] for v in deterministic_violations}
            deduped_llm = [
                v for v in llm_violations
                if v.get("check") not in covered_checks
            ]
            violations = deterministic_violations + deduped_llm

            span.set_attribute("violation_count", len(violations))
            span.set_attribute("shuttle_violation_count", len(shuttle_violations))
            span.set_attribute("derived_violation_count", len(derived_violations))
            span.set_attribute("performance_violation_count", len(performance_violations))
            span.set_attribute("llm_violation_count", len(deduped_llm))
            span.set_attribute("all_pass", len(violations) == 0)
            span.set_attribute("max_gate_count", max_gate_count)
            span.set_attribute("sram_size_kb", sram_size_kb)
            span.set_attribute("needs_bus", needs_bus)
            span.set_attribute("shuttle_constraints_enabled", shuttle_enabled)
            if violations:
                span.set_attribute(
                    "violations",
                    "; ".join(v.get("violation", "") for v in violations[:5]),
                )

            return violations

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(_trace.StatusCode.ERROR, str(e))
            # Still return deterministic violations even if LLM fails
            deterministic_violations = shuttle_violations + derived_violations + performance_violations
            deterministic_violations.append({
                "violation": f"Constraint check LLM failed: {e}. "
                             f"Architecture may have unchecked violations.",
                "category": "structural",
                "check": "llm_error",
                "severity": "warning",
            })
            return deterministic_violations


def _safe_int(value: Any, default: int) -> int:
    """Extract an integer from a value that may be None, str, or numeric."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _walk_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_walk_text(item))
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_walk_text(item))
        return out
    return []


def _extract_sram_budget_kb(ers: dict[str, Any], default: int = 32) -> int:
    """Extract the intended on-chip SRAM budget from PRD/ERS content.

    Older PRD schemas only placed the memory KPI in area/dataflow notes, so
    falling back to a hard 32 KB budget can create false structural failures
    for designs whose stated combined activation+KV budget is 64 KB.
    """
    dataflow = ers.get("dataflow", {}) or {}
    area = ers.get("area_budget", {}) or {}

    for key in (
        "sram_size_kb",
        "sram_budget_kb",
        "onchip_sram_budget_kb",
        "on_chip_sram_budget_kb",
        "combined_sram_budget_kb",
    ):
        if key in dataflow:
            return _safe_int(dataflow.get(key), default)
        if key in area:
            return _safe_int(area.get(key), default)

    text = "\n".join(_walk_text({
        "dataflow": dataflow,
        "area_budget": area,
        "kpis": ers.get("kpis", ers.get("validation_kpis", [])),
        "requirements": ers.get("requirements", []),
    }))
    matches = []
    for match in re.finditer(
        r"(?:<=|less than|no greater than|max(?:imum)?|budget|limit|not exceed)"
        r"[^.\n]{0,80}?(\d+)\s*KB",
        text,
        flags=re.IGNORECASE,
    ):
        matches.append(int(match.group(1)))
    if matches:
        return max(matches)
    return default


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default: dict[str, Any] = {
        "violations": [],
        "reasoning": "",
    }
    result, _ok = parse_llm_json(content, default, context="constraints")
    return result
