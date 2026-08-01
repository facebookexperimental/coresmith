# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR4 guard: EDA prompts run tools via the CLI, not by name.

Two invariants, same spirit as the no-benchmark-refs-in-engine rule:

1. No migrated ``*_llm.md`` / ``tapeout_wrapper_*.md`` / ``skills/*.md`` prompt
   contains a bare tool *invocation* (``yosys -``, ``openroad -``, ``magic -``,
   ``netgen -``), a ``{tool_bin}`` placeholder, or the ``sky130_fd_sc_hd``
   cell-library token. Tool names may still appear in PROSE ("Yosys times out",
   "Magic DRC") -- only invocation-shaped usage is forbidden. ``.legacy.md``
   copies (the pre-migration rollback text) are exempt.

2. Every ``{placeholder}`` a rewritten prompt references is supplied by that
   prompt's graph context dict (a static per-prompt allowlist) UNION the
   deployment-supplied keys merged in by ``eda_prompts.deployment_prompt_context``
   -- so no node hits a KeyError / leaves a field unfilled.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestrator.langgraph.backend_graph import _SAFE_FORMAT_RE

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "langchain" / "prompts"

# Invocation-shaped forbidden patterns (NOT bare prose mentions).
_FORBIDDEN = [
    (re.compile(r"\byosys\s+-"), "bare `yosys -...` invocation (use `$CS tool run_synth`)"),
    (re.compile(r"\bopenroad\s+-"), "bare `openroad -...` invocation (use `$CS tool run_pnr/run_sta`)"),
    (re.compile(r"\{openroad_bin\}"), "{openroad_bin} placeholder (the CLI resolves the binary)"),
    (re.compile(r"\bmagic\s+-"), "bare `magic -...` invocation (use `$CS tool run_drc`)"),
    (re.compile(r"\{magic_bin\}"), "{magic_bin} placeholder (the CLI resolves the binary)"),
    (re.compile(r"\bnetgen\s+-"), "bare `netgen -...` invocation (use `$CS tool run_lvs`)"),
    (re.compile(r"\{netgen_bin\}"), "{netgen_bin} placeholder (the CLI resolves the binary)"),
    (re.compile(r"sky130_fd_sc_hd"), "hardcoded sky130_fd_sc_hd cell-library token"),
]


# The script-adaptation prompts (run under disable_tools=True) also get their
# tool/PDK specifics from {tool_notes}/{pdk_summary} now (PR5), so they must be
# free of hardcoded sky130 cell tokens + bare invocations too.
_ADAPTATION_PROMPTS = (
    "backend_synthesis.md", "backend_pnr.md", "backend_drc.md", "backend_lvs.md",
)


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for pat in ("*_llm.md", "tapeout_wrapper_*.md"):
        files += _PROMPT_DIR.glob(pat)
    files += (_PROMPT_DIR / "skills").glob("*.md")
    files += [_PROMPT_DIR / n for n in _ADAPTATION_PROMPTS]
    return [f for f in sorted(set(files)) if not f.name.endswith(".legacy.md")]


# ---------------------------------------------------------------------------
# Invariant 1: no bare tool invocations in the migrated prompts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", _guarded_files(), ids=lambda p: p.name)
def test_no_bare_tool_invocation(prompt: Path):
    text = prompt.read_text()
    hits = [why for rx, why in _FORBIDDEN if rx.search(text)]
    assert not hits, f"{prompt.name} contains forbidden EDA usage: {hits}"


def test_guard_actually_scans_files():
    names = {p.name for p in _guarded_files()}
    # The migrated verb prompts must be in scope, and legacy copies must NOT.
    assert "backend_synth_llm.md" in names
    assert "tapeout_wrapper_lvs.md" in names
    # The script-adaptation prompts are now in scope too (PR5).
    for n in _ADAPTATION_PROMPTS:
        assert n in names, f"{n} should be guarded"
    assert not any(n.endswith(".legacy.md") for n in names)


def test_legacy_copies_are_exempt_and_hold_the_old_text():
    """A legacy copy still carries the pre-migration invocation -- proof both
    that the guard would catch a regression and that rollback text is real."""
    legacy = _PROMPT_DIR / "backend_synth_llm.legacy.md"
    assert legacy.is_file()
    text = legacy.read_text()
    assert any(rx.search(text) for rx, _ in _FORBIDDEN), \
        "legacy copy should still contain the old bare invocation"


# ---------------------------------------------------------------------------
# Invariant 2: every placeholder is supplied by the graph context
# ---------------------------------------------------------------------------
# Deployment-supplied keys merged in by eda_prompts.deployment_prompt_context.
_DEPLOY_KEYS = {
    "deployment", "pdk_summary", "tool_notes", "std_cell_library",
    "pdk_variant", "cell_spice",
}

# Static per-prompt allowlist = the keys each graph node's context dict fills
# (mirrors backend_graph.py / tapeout_graph.py context construction).
_CALLER_KEYS: dict[str, set[str]] = {
    "backend_synth_llm.md": {
        "design_name", "target_clock_mhz", "period_ns", "liberty_path",
        "output_dir", "input_files", "input_delay_ns", "output_delay_ns",
        "attempt", "prior_failure", "constraints", "result_json_path",
        "sram_macro_directive", "sram_wrapper_lib"},
    "backend_pnr_llm.md": {
        "design_name", "target_clock_mhz", "period_ns", "gate_count",
        "tech_lef", "cell_lef", "liberty_path", "openroad_bin", "netlist_path",
        "sdc_path", "output_dir", "utilization", "density", "margin", "attempt",
        "max_attempts", "prior_failure", "constraints", "result_json_path",
        "tcl_path"},
    "backend_drc_llm.md": {
        "design_name", "magic_rc", "cell_gds", "cell_lef", "tech_lef",
        "magic_bin", "routed_def_path", "output_dir", "attempt",
        "prior_failure", "constraints", "result_json_path", "macro_lefs",
        "macro_names", "has_macros"},
    "backend_lvs_llm.md": {
        "design_name", "netgen_setup", "netgen_bin", "spice_path",
        "pwr_verilog_path", "verilog_path", "output_dir", "attempt",
        "prior_failure", "constraints", "result_json_path"},
    "backend_wrapper_llm.md": {
        "design_name", "target_clock_mhz", "gate_count", "project_root",
        "pnr_verilog_path", "routed_def_path", "gds_path", "spice_path",
        "sdc_path", "spef_path", "submission_dir", "result_json_path"},
    "tapeout_wrapper_synth.md": {
        "liberty_path", "target_clock_mhz", "period_ns", "output_dir",
        "wrapper_rtl_path", "block_netlists", "result_json_path"},
    "tapeout_wrapper_pnr.md": {
        "prd_path", "frd_path", "tech_lef", "cell_lef", "liberty_path",
        "openroad_bin", "target_clock_mhz", "period_ns", "die_width_um",
        "die_height_um", "core_margin_um", "netlist_path", "sdc_path",
        "tcl_path", "output_dir", "attempt", "max_attempts", "prior_failure",
        "pnr_overrides", "result_json_path"},
    "tapeout_wrapper_drc.md": {
        "prd_path", "frd_path", "magic_rc", "cell_gds", "tech_lef", "cell_lef",
        "magic_bin", "routed_def_path", "output_dir", "prior_failure",
        "result_json_path"},
    "tapeout_wrapper_lvs.md": {
        "prd_path", "frd_path", "netgen_setup", "netgen_bin", "spice_path",
        "pwr_verilog_path", "verilog_path", "output_dir", "prior_failure",
        "result_json_path"},
}


def _placeholders(text: str) -> set[str]:
    """Real single-brace {name} placeholders (skips {{ }} escapes + code braces),
    using the engine's own _safe_format regex so extraction matches rendering.

    A ``${name...}`` shell construct (the ``${CORESMITH_CLI:-coresmith}`` alias)
    is NOT a template field -- _safe_format leaves it literal because the name is
    never a context key -- so matches preceded by ``$`` are excluded.
    """
    out: set[str] = set()
    for m in _SAFE_FORMAT_RE.finditer(text):
        if not m.group(1):
            continue
        if m.start() > 0 and text[m.start() - 1] == "$":
            continue  # shell variable, not a template placeholder
        out.add(m.group(1))
    return out


@pytest.mark.parametrize("prompt_name", sorted(_CALLER_KEYS))
def test_every_placeholder_is_supplied(prompt_name: str):
    text = (_PROMPT_DIR / prompt_name).read_text()
    allowed = _CALLER_KEYS[prompt_name] | _DEPLOY_KEYS
    used = _placeholders(text)
    missing = used - allowed
    assert not missing, (
        f"{prompt_name} references placeholders not supplied by its graph "
        f"context or the deployment: {sorted(missing)}")


def test_migrated_prompts_reference_the_cli():
    """Every migrated verb prompt actually invokes the CLI convention."""
    for name in _CALLER_KEYS:
        if name == "backend_wrapper_llm.md":
            continue  # packaging step: no EDA verb invoked
        text = (_PROMPT_DIR / name).read_text()
        assert 'tool run_' in text, f"{name} does not invoke a `$CS tool run_*` verb"
        assert "CORESMITH_CLI" in text, f"{name} does not define the CS alias"


# ---------------------------------------------------------------------------
# eda_prompts: deployment context + rollback flag
# ---------------------------------------------------------------------------
_MIGRATED = [n for n in _CALLER_KEYS] + [
    "tapeout_wrapper_synth.md", "tapeout_wrapper_pnr.md",
    "tapeout_wrapper_drc.md", "tapeout_wrapper_lvs.md",
]


class TestEdaPromptsHelper:
    def test_deployment_context_supplies_tool_notes(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "sky130")
        from orchestrator.pdk.registry import reset_deployment_cache
        reset_deployment_cache()
        from orchestrator.langgraph.eda_prompts import deployment_prompt_context
        ctx = deployment_prompt_context("backend_pnr_llm.md")
        assert ctx.get("pdk_summary")
        assert ctx["tool_notes"], "run_pnr prompt_notes should be non-empty"
        # A prompt that maps to no verb gets an empty tool_notes (not missing).
        assert deployment_prompt_context("backend_wrapper_llm.md")["tool_notes"] == ""

    def test_rollback_flag_selects_legacy(self, monkeypatch):
        from orchestrator.langgraph.eda_prompts import resolve_prompt_path
        monkeypatch.setenv("CORESMITH_TOOL_CLI_PROMPTS", "0")
        p = resolve_prompt_path(_PROMPT_DIR, "backend_synth_llm.md")
        assert p.name == "backend_synth_llm.legacy.md"

    def test_default_flag_selects_migrated(self, monkeypatch):
        from orchestrator.langgraph.eda_prompts import resolve_prompt_path
        monkeypatch.delenv("CORESMITH_TOOL_CLI_PROMPTS", raising=False)
        p = resolve_prompt_path(_PROMPT_DIR, "backend_synth_llm.md")
        assert p.name == "backend_synth_llm.md"

    def test_every_migrated_prompt_has_a_legacy_copy(self):
        for name in _MIGRATED:
            legacy = _PROMPT_DIR / f"{Path(name).stem}.legacy.md"
            assert legacy.is_file(), f"missing rollback copy for {name}"


def test_gen_macro_cli_skips_when_capability_absent(tmp_path, monkeypatch):
    """gen_macro is a CLI verb stub: a deployment lacking it exits 4 (skip),
    never a false green."""
    import types

    from orchestrator.harness import cli_tool
    from orchestrator.pdk.registry import reset_deployment_cache
    monkeypatch.setenv("CORESMITH_DEPLOYMENT", "sky130")
    reset_deployment_cache()
    args = types.SimpleNamespace(
        verb="gen_macro", project_root=str(tmp_path), design="w8d64",
        width=8, depth=64, ports="1rw1r", out_dir=None, timeout_s=None,
        json=False, rtl=None, script=None, netlist=None, sdc=None, gds=None,
        spice=None)
    assert cli_tool.cmd_tool_run(args) == 4
    reset_deployment_cache()
