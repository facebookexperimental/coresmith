"""WP-48: front-end prompts carry no benchmark-, design- or incident-specific guidance."""
from __future__ import annotations

import re
from pathlib import Path

from orchestrator.langgraph import pipeline_graph as pg

P = Path(pg.__file__).resolve().parent.parent / "langchain" / "prompts"
FRONTEND = [
    "block_diagram.md", "uarch_spec_generator.md", "rtl_generator.md",
    "validation_dv.md", "interface_definition.md", "integration_review.md",
    "testbench_generator.md", "contract_audit.md", "sad_spec.md", "frd_spec.md",
    "prd_spec.md", "ers_doc.md", "constraint_check.md", "output_contract_review.md",
]
FORBIDDEN = re.compile(
    r"h\.?264|cavlc|psnr|ax\.?25|\baes\b|ppabench|\(observed|observed live|"
    r"\barm-[a-z]\b|regmap_buffers|modem_controller|qspi_slave_frontend|caravel|openframe|"
    r"user_project_wrapper|wb_clk_i|wb_rst_i|io_oeb|stream_core|stream_tb|video_codec|"
    r"codecv4|macroblock",
    re.I,
)


def _hits(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in FORBIDDEN.finditer(text)})


def test_frontend_prompts_are_general_purpose():
    bad = {}
    for name in FRONTEND:
        hits = _hits((P / name).read_text())
        if hits:
            bad[name] = hits
    assert not bad, bad


def test_skills_are_general_purpose_except_the_protocol_skill():
    bad = {}
    for p in sorted((P / "skills").glob("*.md")):
        if p.name == "qspi_slave_frontend_protocol.md":
            continue
        hits = _hits(p.read_text())
        if hits:
            bad[p.name] = hits
    assert not bad, bad
