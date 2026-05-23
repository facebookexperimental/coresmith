# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HTML Architecture Dashboard Generator.

Uses a Jinja2 template to produce a self-contained HTML dashboard that
presents every architecture artifact (PRD, SAD, FRD, ERS, block diagram,
memory map, clock tree, register spec, uArch specs) on a single page.

The architecture documents (PRD, SAD, FRD, uArch) are rendered directly
from their markdown source via a lightweight markdown-to-HTML converter,
preserving the full document content rather than relying on LLM
summarisation.

The block diagram section includes a Mermaid flowchart generated from
the block/connection JSON data, rendered client-side via mermaid.js.
"""

from __future__ import annotations

import json
import re
from html import escape as _esc
from pathlib import Path


_TMPL_DIR = Path(__file__).resolve().parents[2] / "langchain" / "prompts"
_TMPL_FILE = "dashboard.html.j2"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def generate_dashboard(
    prd_spec: dict | None,
    sad_spec: dict | None,
    frd_spec: dict | None,
    ers_spec: dict | None,
    block_diagram: dict | None,
    memory_map: dict | None,
    clock_tree: dict | None,
    register_spec: dict | None,
    project_root: str = ".",
) -> str:
    """Generate an HTML architecture dashboard via Jinja2 template.

    Returns the raw HTML string ready to write to disk.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.dashboard_doc")

    with tracer.start_as_current_span("generate_dashboard") as span:
        root = Path(project_root)

        prd_md = _read_md(root / "arch" / "prd_spec.md")
        sad_md = _read_md(root / "arch" / "sad_spec.md")
        frd_md = _read_md(root / "arch" / "frd_spec.md")
        ers_md = _read_md(root / "arch" / "ers_spec.md")

        if not prd_md and prd_spec:
            prd_md = _json_to_md(prd_spec, "prd")
        if not sad_md and sad_spec:
            sad_md = sad_spec.get("sad_text", "") or _json_to_md(sad_spec, "sad")
        if not frd_md and frd_spec:
            frd_md = frd_spec.get("frd_text", "") or _json_to_md(frd_spec, "frd")
        if not ers_md and ers_spec:
            ers_md = _json_to_md(ers_spec, "ers")

        uarch_specs = _read_all_uarch(root)

        blocks = (block_diagram or {}).get("blocks", [])
        connections = (block_diagram or {}).get("connections", [])
        mermaid = _blocks_to_mermaid(block_diagram)

        prd_data = (prd_spec or {}).get("prd", prd_spec or {})
        if not isinstance(prd_data, dict):
            prd_data = {}
        design_name, subtitle, metrics, notes = _extract_summary(prd_data, blocks)

        memory_map_html = _md_to_html(_read_md(root / "arch" / "memory_map.md"))
        clock_tree_html = _md_to_html(_read_md(root / "arch" / "clock_tree.md"))
        register_spec_html = _md_to_html(
            _read_md(root / "arch" / "register_spec.md"),
        )

        span.set_attribute("has_prd", bool(prd_md))
        span.set_attribute("has_sad", bool(sad_md))
        span.set_attribute("has_frd", bool(frd_md))
        span.set_attribute("block_count", len(blocks))

        from jinja2 import Environment, FileSystemLoader

        env = Environment(
            loader=FileSystemLoader(str(_TMPL_DIR)),
            autoescape=False,
        )
        template = env.get_template(_TMPL_FILE)

        name_short = design_name
        if len(name_short) > 20:
            name_short = name_short[:18] + "…"

        html = template.render(
            design_name=design_name,
            design_name_short=name_short,
            design_subtitle=subtitle,
            design_notes=notes,
            metrics=metrics,
            mermaid_diagram=mermaid,
            blocks=blocks,
            connections=connections,
            prd_html=_md_to_html(prd_md),
            sad_html=_md_to_html(sad_md),
            frd_html=_md_to_html(frd_md),
            ers_html=_md_to_html(ers_md),
            memory_map=memory_map,
            memory_map_html=memory_map_html,
            clock_tree=clock_tree,
            clock_tree_html=clock_tree_html,
            register_spec=register_spec,
            register_spec_html=register_spec_html,
            uarch_specs=uarch_specs,
        )
        span.set_attribute("html_length", len(html))
        return html


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


def _extract_summary(
    prd: dict,
    blocks: list[dict],
) -> tuple[str, str, list[dict], str]:
    """Pull key metrics from the PRD for the summary cards."""

    title = prd.get("title", "ASIC Design")
    summary_text = prd.get("summary", "")

    tech = prd.get("target_technology", {})
    pdk = tech.get("pdk", "—")
    process = tech.get("process_nm", "—")

    sf = prd.get("speed_and_feeds", {})
    clock = sf.get("target_clock_mhz", "—")

    area = prd.get("area_budget", {})
    gate_budget = area.get("max_gate_count", "—")

    power = prd.get("power_budget", {})
    power_mw = power.get("total_power_mw", "—")

    df = prd.get("dataflow", {})
    protocol = df.get("bus_protocol", "—")
    data_w = df.get("data_width_bits", "—")

    total_gates = sum(b.get("estimated_gates", 0) for b in blocks)

    subtitle_parts = []
    if pdk != "—":
        subtitle_parts.append(f"{pdk} {process} nm")
    if clock != "—":
        subtitle_parts.append(f"{clock} MHz target")
    if protocol != "—":
        subtitle_parts.append(protocol)
    subtitle = " · ".join(subtitle_parts) if subtitle_parts else summary_text[:80]

    metrics = []
    if pdk != "—":
        metrics.append({"label": f"PDK / {process} nm", "value": pdk})
    if clock != "—":
        metrics.append({"label": "Target Clock", "value": f"{clock} MHz"})
    if gate_budget != "—":
        metrics.append({"label": "Gate Budget", "value": f"{gate_budget:,}" if isinstance(gate_budget, int) else str(gate_budget)})
    if total_gates:
        metrics.append({"label": "Est. Gates", "value": f"~{total_gates:,}"})
    metrics.append({"label": "Blocks", "value": str(len(blocks))})
    if power_mw != "—":
        metrics.append({"label": "Power Budget", "value": f"{power_mw} mW"})
    if data_w != "—":
        metrics.append({"label": "Data Width", "value": f"{data_w}-bit"})
    if protocol != "—":
        metrics.append({"label": "Bus Protocol", "value": protocol})

    return title, subtitle, metrics, ""


# ---------------------------------------------------------------------------
# Mermaid diagram generation
# ---------------------------------------------------------------------------


def _blocks_to_mermaid(diagram: dict | None) -> str:
    """Generate a Mermaid flowchart from block diagram JSON."""
    if not diagram:
        return ""

    blocks = diagram.get("blocks", [])
    connections = diagram.get("connections", [])
    if not blocks:
        return ""

    lines = ["graph LR"]

    subsystems: dict[str, list[dict]] = {}
    for b in blocks:
        sub = b.get("subsystem", "") or ""
        subsystems.setdefault(sub, []).append(b)

    for sub, sub_blocks in subsystems.items():
        if sub:
            label = sub.replace("_", " ").title()
            lines.append(f"  subgraph {_mermaid_id(sub)} [\"{label}\"]")
            for b in sub_blocks:
                lines.append(_mermaid_node(b, indent=4))
            lines.append("  end")
        else:
            for b in sub_blocks:
                lines.append(_mermaid_node(b, indent=2))

    for conn in connections:
        src = _mermaid_id(conn.get("from", ""))
        dst = _mermaid_id(conn.get("to", ""))
        iface = conn.get("interface", "")
        width = conn.get("data_width", "")
        bus = conn.get("bus_name", "")
        parts = []
        if width:
            parts.append(f"{width}b")
        if iface:
            parts.append(iface)
        if bus:
            parts.append(f"via {bus}")
        label = " ".join(parts)
        if label:
            lines.append(f"  {src} -->|\"{label}\"| {dst}")
        else:
            lines.append(f"  {src} --> {dst}")

    return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Sanitise a block name for use as a Mermaid node ID."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _mermaid_node(block: dict, indent: int = 2) -> str:
    """Format a single Mermaid node definition."""
    name = _mermaid_id(block["name"])
    display = block["name"]
    gates = block.get("estimated_gates")
    tier = block.get("tier")
    extras = []
    if gates:
        extras.append(f"~{gates:,}g")
    if tier:
        extras.append(f"T{tier}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    pad = " " * indent
    return f'{pad}{name}["{display}{suffix}"]'


# ---------------------------------------------------------------------------
# Markdown ↔ HTML conversion
# ---------------------------------------------------------------------------


def _md_to_html(text: str) -> str:
    """Convert markdown to HTML for embedding in the dashboard.

    Handles headings, bold, italic, inline code, lists, tables,
    fenced code blocks, and paragraphs.  Not a full CommonMark
    implementation -- just enough for architecture documents.
    """
    if not text or not text.strip():
        return ""

    result: list[str] = []
    in_code = False
    in_ul = False
    in_ol = False
    in_table = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith("```"):
            if in_code:
                result.append("</code></pre>")
                in_code = False
            else:
                result.append('<pre class="code-block"><code>')
                in_code = True
            continue
        if in_code:
            result.append(_esc(line))
            continue

        # Empty line closes open containers
        if not stripped:
            if in_ul:
                result.append("</ul>")
                in_ul = False
            if in_ol:
                result.append("</ol>")
                in_ol = False
            if in_table:
                result.append("</tbody></table>")
                in_table = False
            continue

        # Headings
        hm = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if hm:
            _close_list(result, in_ul, in_ol)
            in_ul = in_ol = False
            lvl = len(hm.group(1))
            result.append(f"<h{lvl}>{_inline(hm.group(2))}</h{lvl}>")
            continue

        # Tables
        if "|" in stripped and (
            stripped.startswith("|") or re.match(r"\S+\s*\|", stripped)
        ):
            cells = [c.strip() for c in stripped.split("|")]
            cells = [c for c in cells if c != ""]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            if not in_table:
                result.append("<table><thead><tr>")
                for c in cells:
                    result.append(f"<th>{_inline(c)}</th>")
                result.append("</tr></thead><tbody>")
                in_table = True
            else:
                result.append("<tr>")
                for c in cells:
                    result.append(f"<td>{_inline(c)}</td>")
                result.append("</tr>")
            continue

        # Unordered list
        um = re.match(r"^[-*]\s+(.+)$", stripped)
        if um:
            if not in_ul:
                result.append("<ul>")
                in_ul = True
            result.append(f"<li>{_inline(um.group(1))}</li>")
            continue

        # Ordered list
        om = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if om:
            if not in_ol:
                result.append("<ol>")
                in_ol = True
            result.append(f"<li>{_inline(om.group(2))}</li>")
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}$", stripped):
            result.append("<hr>")
            continue

        # Paragraph
        result.append(f"<p>{_inline(stripped)}</p>")

    if in_ul:
        result.append("</ul>")
    if in_ol:
        result.append("</ol>")
    if in_table:
        result.append("</tbody></table>")
    if in_code:
        result.append("</code></pre>")

    return "\n".join(result)


def _close_list(result: list[str], in_ul: bool, in_ol: bool) -> None:
    if in_ul:
        result.append("</ul>")
    if in_ol:
        result.append("</ol>")


def _inline(text: str) -> str:
    """Convert inline markdown: bold, italic, code, links."""
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"_([a-zA-Z].*?[a-zA-Z])_", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _read_md(path: Path, max_chars: int = 80_000) -> str:
    """Read a markdown file, returning empty string on failure."""
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return text[:max_chars] if len(text) > max_chars else text
    except OSError:
        pass
    return ""


def _read_all_uarch(root: Path) -> list[dict]:
    """Read all uArch spec markdown files, returning list of {name, html}."""
    from orchestrator.architecture.state import ARCH_DOC_DIR

    specs_dir = root / ARCH_DOC_DIR / "uarch_specs"
    if not specs_dir.is_dir():
        return []

    specs: list[dict] = []
    for spec_file in sorted(specs_dir.glob("*.md")):
        try:
            md = spec_file.read_text(encoding="utf-8")
            specs.append({
                "name": spec_file.stem,
                "html": _md_to_html(md),
            })
        except OSError:
            continue
    return specs


def _json_to_md(data: dict, key: str) -> str:
    """Fallback: render a JSON dict as a readable markdown-ish string."""
    inner = data.get(key, data)
    return f"```json\n{json.dumps(inner, indent=2)}\n```"
