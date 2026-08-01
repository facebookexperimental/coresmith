# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
RTLGeneratorAgent -- converts Python DSP models to Verilog RTL.

Reads a Python source file implementing a signal processing block (e.g., LFSR
scrambler, Reed-Solomon encoder, FFT butterfly), understands the algorithm,
and generates synthesizable Verilog-2005 with AXI-Stream interfaces.

All invocations are traced via OpenTelemetry for observability and evaluation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator.langchain.prompts.skills import load_skill_strict as _load_skill_strict
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

from .coresmith_llm import DEFAULT_MODEL, ClaudeLLM

_tracer = trace.get_tracer(__name__)

# Load system prompt from template file, fall back to inline
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "rtl_generator.md"
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = """\
You are an expert digital design engineer specializing in converting Python
signal processing models into synthesizable Verilog-2005 RTL for ASIC
implementation on the SkyWater Sky130 130nm process.

RULES:
1. Output ONLY valid Verilog-2005 (no SystemVerilog constructs).
2. Use AXI-Stream (tdata/tvalid/tready/tlast) for data interfaces.
3. Use synchronous active-low reset (rst_n).
4. Use a single clock domain (clk).
5. All arithmetic must be fixed-point -- no floating point.
6. Use explicit bit widths on all signals. No implicit widths.
7. Include a module header comment with: block name, description, I/O ports.
8. Registers must have reset values.
9. FSMs must use localparam for state encoding.
10. No latches -- every conditional must have an else clause.
11. Combinational logic in always @(*) blocks, sequential in always @(posedge clk).
12. Target: fully synthesizable by Yosys for Sky130.
13. Exactly ONE implementation per module. NEVER put two versions of the logic
    in one module behind a `ifdef/`ifndef/`elsif (e.g. the real algorithm under
    `ifndef SYNTHESIS and a mock under `else) -- DV verifies one branch while
    synth/backend build the other (DIFFERENT HARDWARE, so DV proves nothing).
    Conditional compilation may ONLY guard non-functional debug/trace/assertion
    code ($display, $dumpvars, SVA assert). If simulation needs behavior synth
    gets from a macro, that split lives ONLY inside the provided cs_* wrapper
    library -- never in your module. A deterministic gate rejects violations.

AXI-STREAM OUTPUT FSM -- CRITICAL:
When producing output on an AXI-Stream master port, you MUST follow this
two-phase pattern to avoid the "valid self-cancellation" bug:

  WRONG (valid is set and cleared in the same combinational pass):
    ST_OUTPUT: begin
        m_tvalid_next = 1'b1;          // set valid...
        if (m_tready)                   // ...but tready is already 1...
            m_tvalid_next = 1'b0;      // ...so valid is immediately cleared!
    end
    // Result: m_tvalid_reg NEVER becomes 1. Deadlock.

  CORRECT (set valid, wait one cycle for handshake):
    ST_OUTPUT: begin
        m_tvalid_next = 1'b1;          // assert valid
        if (m_tvalid_reg && m_tready)   // handshake on REGISTERED valid
            m_tvalid_next = 1'b0;      // clear after transfer
            state_next = ST_IDLE;
        end
    end
    // Result: valid rises for at least 1 cycle, handshake completes.

  SIMPLEST (registered output, always correct):
    always @(posedge clk)
        if (!rst_n) m_tvalid <= 0;
        else if (produce_data) m_tvalid <= 1;
        else if (m_tvalid && m_tready) m_tvalid <= 0;

When converting Python to Verilog:
- Map numpy arrays to register files or SRAM. **Any storage >= 16384 bits AND
  >= 256 words deep is an SRAM, NOT a flop array: instantiate the generic wrapper
  `cs_sram_1rw #(.WIDTH(w), .DEPTH(d)) u_name (.clk, .ce, .we, .addr, .wdata,
  .rdata)` (or `cs_sram_1rw1r` for a 2-read-port memory). NEVER write a raw
  `reg [w-1:0] mem [0:d-1];` for storage that big -- the lint gate will reject
  it.** The `cs_sram` wrapper is PROVIDED BY THE TOOLFLOW (a shared library that
  is auto-included in lint/sim/synth) -- you ONLY *instantiate* it; do NOT
  define, redeclare, or paste the `module cs_sram_1rw`/`cs_sram_1rw1r` body into
  your block file (that causes a Verilator MODDUP duplicate-module error). The
  wrapper is behavioral in simulation and is replaced by an OpenRAM/sky130 SRAM
  macro at synthesis & backend, so it costs ~0 flip-flops. Do NOT name a
  specific macro; parametrize the wrapper and the flow resolves the geometry.
  Smaller register files (< 16384 bits or < 256 deep, e.g. a 16-deep line buffer
  or a same-cycle multi-read scan array) stay as plain `reg` arrays.
- Map Python loops to a REGISTERED FSM/datapath (sequentialize the body over
  cycles). Unroll combinationally ONLY when one iteration's arithmetic fits a
  single clock period -- NEVER unroll a multi-op search/transform/accumulation
  into one combinational cloud (functionally correct but UNSYNTHESIZABLE: the
  synth gate times out). If the uArch spec names a pipeline depth, realize each
  stage as a registered always block, not one always @(*). See pipeline_contract.
- Map dictionary lookups to ROM/LUT.
- Map floating-point math to fixed-point (specify Q format in comments).
- Handle variable-length data with valid/ready handshaking.
- A ready/valid transfer is exactly `valid && ready` sampled on the clock edge.
  Do not qualify the handshake with a registered copy of `ready`, a previous
  cycle's ready, or a requirement that ready stay high for two cycles. If a
  registered output token is held valid, a one-cycle `ready` pulse must retire
  exactly one token and advance state once.

If the previous attempt failed, the error will be provided. Fix the specific
issue while maintaining correctness.

Output format:
1. The complete Verilog module (one module per response).
2. After the module, a JSON block with port information:
   ```json
   {{"module_name": "...", "ports": {{"clk": "input", ...}}}}
   ```
"""

# Hand the RTL implementer the SAME pipeline-synthesizability discipline the
# uArch spec author has. The codec RD-search failure was an RTL-fidelity gap:
# the spec correctly described an N-stage pipeline, but the RTL generator --
# which never saw this skill or the PDK budget -- collapsed the datapath into
# one combinational always-block cloud that walls the synth gate at >600s.
_SKILLS_TEXT = _load_skills("pipeline_contract", "verify_in_context")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skill (synthesizable-pipeline discipline -- MANDATORY)"
        + "\n\n"
        + _SKILLS_TEXT
    )

# port_naming is ALWAYS inline (it is ~2 KB). The one rule that has no cheap
# recovery path: a collapsed `<channel>_<field>` name is not caught until the
# deterministic pre-sim conformance gate, and costs a whole regeneration. It
# lived in NO prompt while the RTL generator was told to transcribe the golden
# model's (collapsed) port list "byte-exact" -- see the AUTHORITATIVE PORT
# NAMES table injected into every RTL user message.
_PORT_NAMING_SKILL = _load_skill_strict("port_naming")
_LATCHED_CTRL_SKILL = _load_skill_strict("latched_control_decisions")
SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\n\n# Reference Skill (canonical port naming -- MANDATORY)\n\n"
    + _PORT_NAMING_SKILL
    + "\n\n"
    + _LATCHED_CTRL_SKILL
)

# QSPI-slave frontend / IO-subsystem blocks own the external chassis bus boundary
# and keep shipping protocol-INCOMPLETE code (dropped cmd 0x05 read_status, short
# read dummy, read launched a nibble early) because the bus protocol is re-derived
# per design. Match such a block by name/description so the protocol-completeness
# skill is injected ONLY for the block that owns the bus boundary, not every DSP
# block.
_QSPI_FRONTEND_TOKENS = (
    "qspi", "frontend", "front_end", "io_subsystem", "io_sub", "iosub",
    "io_ctrl", "io_bridge", "host_if", "hostif", "host_interface", "bus_if",
    "gpio_ctrl", "io_frontend", "spi_slave", "regmap", "reg_map",
)


def _is_qspi_frontend_block(block_name: str, description: str = "") -> bool:
    """True when the block being authored owns the QSPI-slave bus boundary."""
    hay = f"{block_name} {description}".lower()
    return any(tok in hay for tok in _QSPI_FRONTEND_TOKENS)


def _pdk_budget_fragment(project_root: str = "") -> str:
    """Characterized PDK timing budget for the RTL implementer (gated).

    Returns '' unless ``CORESMITH_PDK_CHAR`` is on and the arithmetic model is
    cached.  This is the SAME per-op delay budget the uArch spec author
    consumes -- now also handed to the RTL generator so it sizes its registered
    pipeline stages to the real per-op delays instead of collapsing the
    datapath into one unsynthesizable combinational cloud. Priced at the run's
    real target clock (resolved from the run dir), not a hardcoded 50 MHz.
    """
    try:
        from orchestrator.langgraph import pdk_characterize as _pdkc
        from orchestrator.langgraph.latency_audit import (
            resolve_target_clock_mhz as _clk,
        )
        from orchestrator.langgraph.pipeline_scheduler import (
            pdk_budget_section as _budget,
        )

        if _pdkc.stage_enabled() and _pdkc.is_characterized():
            mhz = _clk(project_root) if project_root else 50.0
            return _budget(mhz) or ""
    except Exception:  # noqa: BLE001 - best-effort, never block RTL generation
        return ""
    return ""


def _throughput_contract_fragment(project_root: str, block_name: str) -> str:
    """The block's DECLARED cycles/op + II as a HARD RTL constraint (v3).

    The measured-throughput gate rejects an RTL whose measured cyc/op exceeds
    the uArch-declared §6.1 number x 1.1, so the RTL implementer must SEE that
    number up front (the plan->RTL drift the AES serial key schedule proved --
    the RTL worker had no throughput language at all). Reuses the SAME perf
    parser the roofline/gate use. Returns '' when the block declares no `perf`
    block (nothing to enforce) or on any error -- best-effort, never blocks RTL
    generation.
    """
    try:
        from orchestrator.langgraph import throughput_gate as _tg
        declared = _tg.declared_cyc_per_op(project_root, block_name)
        if declared is None or declared <= 0:
            return ""
        peak = _tg.roofline_peak_cyc_per_op(project_root, block_name)
        binding = _tg.binding_constraint_text(project_root, block_name)
        thr = _tg.threshold_for(declared)
        lines = [
            f"DECLARED cycles/op (uArch §6.1) = {declared:g}. Your RTL WILL be "
            f"cycle-measured in DV and REJECTED if measured cyc/op > "
            f"{thr:g} (declared x {_tg.THRESHOLD_RATIO}).",
        ]
        if peak is not None:
            lines.append(
                f"Roofline PEAK cyc/op = {peak:g} (the fully-pipelined floor). "
                "Aim for the declared number; do not serialize below it.")
        if binding:
            lines.append(f"Binding constraint: {binding}.")
        lines.append(
            "Realize the DECLARED schedule: if a stage is declared "
            "word-parallel / II=1, build the parallel lanes -- do NOT serialize "
            "it through one shared resource, and do NOT wrap a compile-time-"
            "static sequence in a per-iteration REQUEST/RESPONSE handshake "
            "(pre-stage locally; see rule 18 + the srdy_drdy skill).")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - best-effort, never block RTL generation
        return ""


def _prompt_slim_enabled() -> bool:
    """CORESMITH_PROMPT_SLIM (default ON): swap giant inlined dumps (hw-golden
    source, full contract JSON) for a path/head + a `coresmith` pull command,
    since agents now have the verify/query CLI on PATH. Set to 0 to restore the
    full inlined prompts (the pre-B4 behavior)."""
    return os.environ.get("CORESMITH_PROMPT_SLIM", "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _read_hw_golden_source(
    project_root: str, rel_path: str, max_bytes: int = 80000
) -> str:
    """Read the per-block Amaranth hardware-golden source for INLINING into RTL
    prompt.

    Path-only references are the documented anti-pattern: the interface
    contracts are inlined for exactly this reason ("go read the file is not
    enough -- inlining forces it into the generator's context window"). The
    byte-exact lowering target must be forced into context, not left to a
    disk-read the agent may skim or skip -- the failure mode where the RTL
    re-approximated the model with a cheap heuristic while the model sat unread.
    Best-effort: returns '' on any failure (never blocks RTL generation).
    """
    if not rel_path:
        return ""
    try:
        rp = Path(rel_path)
        p = rp if rp.is_absolute() else (Path(project_root or ".") / rp)
        text = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text) > max_bytes:
        text = (
            text[:max_bytes]
            + "\n# ... (truncated for prompt size; read the full file from disk)\n"
        )
    return text


def _recover_codex_artifact(
    project_root: str | None, rel_path: str, min_mtime: float = 0.0
) -> str:
    """Recover a file written inside a Codex per-call workdir.

    ``min_mtime`` (C12): only artifacts written at/after this timestamp are
    eligible -- codex-call-* dirs from PREVIOUS calls persist on disk, and an
    unfiltered newest-first glob resurrects a prior call's output as if this
    call produced it.
    """
    if not project_root or not rel_path:
        return ""
    root = Path(project_root)
    try:
        candidates = sorted(
            root.glob(f"codex-call-*/{Path(rel_path).as_posix()}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return ""
    for candidate in candidates:
        try:
            if min_mtime and candidate.stat().st_mtime < min_mtime:
                continue
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if re.search(r"^\s*module\s+\w+", text, re.MULTILINE):
            return text + "\n"
    return ""


def _contract_port_table_fragment(project_root: str, block_name: str) -> str:
    """The block's AUTHORITATIVE port names, derived from the frozen contract.

    Reuses the conformance gate's own derivation
    (``contract_conformance.contract_port_rows``) so the names the generator is
    told to use are byte-identical to the names the gate demands. Without this
    the RTL prompt carried only the golden model's COLLAPSED port list under a
    "transcribe it EXACTLY (byte-exact)" directive, and the contract's port
    table appeared in no prompt at all -- which is how a run lost 90 minutes to
    `host_write_enable` where the contract said `host_write_write_enable`.

    Best-effort: returns '' when there is no contract edge for this block or on
    any error (never blocks RTL generation -- the gate still catches it).
    """
    if not project_root or not block_name:
        return ""
    try:
        from orchestrator.langgraph.contract_conformance import (
            format_contract_port_table,
        )
        return format_contract_port_table(project_root, block_name)
    except Exception:  # noqa: BLE001 - best-effort, never block RTL generation
        return ""


def build_user_message(
    block_name: str,
    description: str = "",
    attempt: int = 1,
    rtl_target: str = "",
    python_source_path: str = "",
    reference_is_hw_golden: bool = False,
    project_root: str = "",
    rtl_language: str = "Verilog-2005",
) -> str:
    """Assemble the RTL generator's user message (the production constructor).

    Split out of :meth:`RTLGeneratorAgent.generate` so the assembled prompt can
    be asserted on directly, the way the other prompt builders in this repo
    are.
    """
    parts = [
        f"Block name: {block_name}",
        f"Description: {description}",
        f"Attempt: {attempt}",
        "",
        "## Working Files",
        "Read these files to understand the design:",
        f"- uArch Spec: arch/uarch_specs/{block_name}.md",
        "- ERS: arch/ers_spec.md",
        f"- Constraints: .coresmith/blocks/{block_name}/constraints.json",
        (
            f"- Hardware Golden Model (Amaranth): {python_source_path}"
            if reference_is_hw_golden
            else f"- Golden Model: {python_source_path}"
        ),
        "- Block Diagram: .coresmith/block_diagram.json (for interface context)",
        "- Interface Contracts: .coresmith/interface_contracts.json "
        "(canonical bit-level edge contracts — see inline excerpt below)",
    ]

    if reference_is_hw_golden:
        # Step 1 of the microarchitecture restructure: the RTL is a
        # *lowering* of the per-block Amaranth hardware golden, not an
        # independent re-transcription of a float reference. This is the
        # fix for "the RTL re-derives (and re-breaks) the hardware
        # decisions the model already paid to make."
        parts += [
            "",
            "## Reference is the HARDWARE GOLDEN — lower it, do not re-derive",
            "The golden model above is the per-block Amaranth HARDWARE model, "
            "NOT a floating-point reference. It is already in hardware "
            "semantics: fixed-point arithmetic, resolved feedback/state, "
            "and any performance derating already applied. Your job is to "
            "LOWER it to RTL faithfully — the RTL must be FUNCTIONALLY "
            "BYTE-EXACT to this model. Do NOT re-derive the algorithm from "
            "a floating-point reference; do NOT change quantization, scan "
            "order, rounding, saturation, or datapath widths; do NOT "
            "'improve' or re-approximate the math. Transcribe its "
            "arithmetic and control exactly. You MAY choose pipelining, "
            "encoding, and resource sharing for PPA, but every produced "
            "value must match the model bit-for-bit. This is about "
            "BEHAVIOR: the model's PORT NAMES are not authoritative — see "
            "the AUTHORITATIVE PORT NAMES table below.",
        ]

        # Inline the hardware-golden SOURCE (not just its path). The
        # path-only reference let the RTL generator skim/skip the model
        # and emit a cheap re-approximation that lints clean; inlining
        # forces the byte-exact target into context, the same fix the
        # interface contracts already use below.
        _hw_src = _read_hw_golden_source(project_root, python_source_path)
        if not _hw_src.strip():
            # D1: the prompt three paragraphs up has just told the
            # generator "the golden model above is the per-block Amaranth
            # HARDWARE model ... the RTL must be FUNCTIONALLY BYTE-EXACT
            # to this model". If the file cannot be read, that sentence
            # is false and the generator is being asked to be byte-exact
            # to nothing -- it will re-derive the algorithm, which is the
            # precise failure the hardware-golden flag exists to stop.
            # Reading "" and carrying on was a silent fail-open.
            raise RuntimeError(
                f"RTL generation for '{block_name}': the run promised a "
                f"HARDWARE GOLDEN ({python_source_path}) as the "
                "byte-exact lowering target (CORESMITH_RTL_FROM_HW_GOLDEN"
                "), but it is empty or unreadable. Refusing to ask for a "
                "byte-exact lowering of nothing -- the generator would "
                "re-derive the algorithm and the equivalence gate would "
                "then fail on RTL nobody could explain. Regenerate the "
                "block model, or turn the flag off to fall back to the "
                "reference golden."
            )
        if _hw_src and _prompt_slim_enabled():
            # B4 prompt-slim: path + head instead of the full inline.
            # The equivalence gate (with a fresh seed) catches any
            # re-derivation, so the byte-exact target need not be dumped.
            _head = "\n".join(_hw_src.splitlines()[:40])
            parts += [
                "",
                "## HARDWARE GOLDEN MODEL — transcribe it EXACTLY (byte-exact)",
                f"The hardware golden is `{python_source_path}` (READ THE "
                "FULL FILE; first 40 lines shown for orientation). The RTL "
                "must be functionally BYTE-EXACT to it: same algorithm, "
                "mode/branch selection, arithmetic, quantization, scan "
                "order, rounding, saturation, and datapath widths. Do NOT "
                "substitute a heuristic or re-derive from a float reference "
                "-- an RTL-vs-model equivalence gate "
                "(`\"${CORESMITH_CLI:-coresmith}\" verify rtl <block>`) "
                "checks this with a FRESH seed before the block "
                "is accepted. Byte-exact applies to BEHAVIOR, not to the "
                "model's port identifiers: take every port NAME from the "
                "AUTHORITATIVE PORT NAMES table below.",
                "```python",
                _head,
                "```",
            ]
        elif _hw_src:
            parts += [
                "",
                "## HARDWARE GOLDEN MODEL SOURCE — transcribe this EXACTLY",
                f"Below is `{python_source_path}` inlined. The RTL must be "
                "functionally BYTE-EXACT to THIS code: same algorithm, "
                "mode/branch selection, arithmetic, quantization, scan "
                "order, rounding, saturation, and datapath widths. Do NOT "
                "substitute a simpler heuristic, gradient/edge "
                "approximation, or any re-derivation — every produced "
                "value must match it bit-for-bit (an RTL-vs-model "
                "equivalence gate checks this before the block is "
                "accepted). Byte-exact applies to BEHAVIOR, not to the "
                "model's port identifiers: take every port NAME from the "
                "AUTHORITATIVE PORT NAMES table below.",
                "```python",
                _hw_src,
                "```",
            ]

    # THE NAMING AUTHORITY. Derived from the frozen contract via the same
    # machinery the pre-sim conformance gate uses, so what the generator is
    # shown and what the gate demands cannot drift apart.
    _port_table = _contract_port_table_fragment(project_root, block_name)
    if _port_table:
        parts.append(_port_table)
        parts.append(
            "\n" + _constraint_precedence_line()
        )

    # Inject the canonical contract slice for this block directly
    # into the prompt. The v7 autopilot run proved that telling the
    # agent "go read interface_contracts.json" is not enough — the
    # RTL generator routinely ignored the file's bootstrap_policy.
    # Inlining the relevant edges forces the contract into the
    # generator's context window.
    from .contract_lookup import (
        format_block_contracts_prompt,
        format_block_contracts_prompt_slim,
        load_block_contracts,
    )
    _contracts_view = load_block_contracts(project_root, block_name)
    # B4 prompt-slim: emit the bootstrap policy + a `coresmith contracts
    # <block>` pointer instead of the full edge JSON dump.
    if _prompt_slim_enabled():
        _contracts_fragment = format_block_contracts_prompt_slim(
            block_name, _contracts_view
        )
    else:
        _contracts_fragment = format_block_contracts_prompt(
            block_name, _contracts_view
        )
    if _contracts_fragment:
        parts.append(_contracts_fragment)

    # PDK arithmetic timing budget (gated; best-effort) -- the real
    # per-op delays so the RTL sizes each registered pipeline stage
    # instead of emitting a single-cycle combinational cloud. Same
    # budget the uArch spec author consumed; honored here at RTL time.
    _budget_text = _pdk_budget_fragment(project_root)
    if _budget_text:
        parts.append(
            "\n--- PDK TIMING BUDGET (size each REGISTERED stage to "
            "these per-op delays; the chained combinational delay in "
            "any one stage must not exceed the clock period) ---\n"
            + _budget_text
        )

    # Per-block PIPELINE STAGE MAP, audited from the Amaranth model's
    # declared STAGE_BUDGET: each named stage -> a registered boundary
    # with a known per-cycle op budget. This is the structural contract
    # whose absence let the codec RD-search collapse into one comb cloud.
    try:
        from orchestrator.langgraph.latency_audit import (
            stage_map_fragment as _stage_map,
        )
        _sm = _stage_map(project_root, block_name)
        if _sm:
            parts.append(
                "\n--- PIPELINE STAGE MAP (from the model's audited "
                "latency budget; realize EACH stage as a registered "
                "always @(posedge clk) boundary) ---\n" + _sm
            )
    except Exception:  # noqa: BLE001 - best-effort, never block RTL gen
        pass

    # THROUGHPUT CONTRACT (v3): the block's DECLARED cyc/op + II as a
    # HARD constraint. Delivery-time DV cycle-measures the RTL and
    # rejects it above declared x 1.1 -- so the target the AES worker
    # never saw is now in the RTL generator's context.
    _thr_text = _throughput_contract_fragment(project_root, block_name)
    if _thr_text:
        parts.append(
            "\n--- THROUGHPUT CONTRACT (your RTL is cycle-measured in "
            "DV; exceeding declared x 1.1 is an automatic rejection -- "
            "see system-prompt rule 18) ---\n" + _thr_text
        )

    if attempt > 1:
        parts.extend([
            f"- Previous Error: .coresmith/blocks/{block_name}/previous_error.txt",
            f"- Existing RTL: {rtl_target} (use Edit to fix incrementally if possible)",
        ])

    parts.extend([
        "",
        "## Output",
        f"Write the complete synthesizable {rtl_language} module to: {rtl_target}",
        "Keep reset, handshakes, state, sideband metadata, error flags, "
        "and pipeline boundary signals named and observable so the "
        "downstream Verilator VCD can be audited with WaveKit.",
        "",
    ])

    if attempt > 1:
        parts.append(
            "This is a RETRY. Read the previous error and the existing RTL. "
            "If the fix is surgical, use the Edit tool to modify the existing "
            "RTL in-place. Only regenerate from scratch if the design is "
            "fundamentally wrong."
        )
    else:
        parts.append(
            "Read the uArch spec and golden model, then generate the "
            "complete Verilog module and write it to the output path."
        )

    return "\n".join(parts)


def _constraint_precedence_line() -> str:
    """Naming precedence for the accumulated constraints.json ('' on error)."""
    try:
        from orchestrator.langgraph.contract_conformance import (
            CONSTRAINT_PRECEDENCE_LINE,
        )
        return CONSTRAINT_PRECEDENCE_LINE
    except Exception:  # noqa: BLE001
        return ""


class RTLGeneratorAgent:
    """Agent for Python-to-Verilog RTL generation.

    DISK-FIRST: Tools are enabled so the agent reads all input files
    (uArch spec, ERS, constraints, golden model, previous error) directly
    from disk and writes the Verilog output to disk.  On retry (attempt > 1),
    the agent can use Edit to incrementally fix existing RTL instead of
    regenerating from scratch.
    """

    _DEFAULT_PROCESS = "SkyWater Sky130 130nm"
    _DEFAULT_RTL_LANG = "Verilog-2005"
    _DEFAULT_SYNTH_TOOL = "Yosys"
    _DEFAULT_CONSTRAINTS = "No tri-state buffers, no async resets, no latches (Sky130 limitations)."

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        # RTL generation is the load-bearing step, so it defaults to the
        # project's DEFAULT_MODEL (Opus). Override via CORESMITH_MODEL or the
        # model= kwarg if a cheaper model is acceptable for a given run.
        model = model or DEFAULT_MODEL
        # 1800s default; bump via CORESMITH_RTL_TIMEOUT env var for complex blocks
        # like CPUs / multi-stage pipelines that need more agent turns to write.
        self.llm = ClaudeLLM(
            model=model,
            timeout=int(os.environ.get("CORESMITH_RTL_TIMEOUT", "1800")),
        )

    async def generate(
        self,
        block_name: str,
        description: str = "",
        attempt: int = 1,
        rtl_target: str = "",
        python_source_path: str = "",
        reference_is_hw_golden: bool = False,
        project_root: str = "",
        target_process: str = "",
        rtl_language: str = "",
        synthesis_tool: str = "",
        process_constraints: str = "",
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Generate Verilog RTL -- agent reads all context from disk.

        Args:
            block_name: Name of the hardware block
            description: Human-readable description
            attempt: Current attempt number
            rtl_target: Relative path to write Verilog (e.g. rtl/foo/bar.v)
            python_source_path: Relative path to Python golden model
            project_root: Path to project root

        Returns:
            Dict with keys: rtl_path (or error)
        """
        block_title = block_name.replace("_", " ").title()
        retry_label = f" - Retry #{attempt - 1}" if attempt > 1 else ""
        span_name = f"RTL Generator [{block_title}]{retry_label}"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("attempt", attempt)

            _proc = target_process or self._DEFAULT_PROCESS
            _lang = rtl_language or self._DEFAULT_RTL_LANG
            _tool = synthesis_tool or self._DEFAULT_SYNTH_TOOL
            _pcon = process_constraints or self._DEFAULT_CONSTRAINTS

            user_message = build_user_message(
                block_name=block_name,
                description=description,
                attempt=attempt,
                rtl_target=rtl_target,
                python_source_path=python_source_path,
                reference_is_hw_golden=reference_is_hw_golden,
                project_root=project_root,
                rtl_language=_lang,
            )

            # NOTE: use explicit placeholder substitution (NOT str.format) so
            # literal braces in prompt code examples -- e.g. the anti-memorization
            # skill's Verilog `key = {mb_cols, mb_rows, qp, mb_y, mb_x};` or
            # concatenations `{ZERO_COEFF_LEVELS, ...}` -- are not misparsed as
            # format fields (which raised KeyError and aborted RTL generation).
            # (engine fix, 2026-06-21)
            system_prompt = SYSTEM_PROMPT
            for _ph, _val in (
                ("{target_process}", _proc),
                ("{rtl_language}", _lang),
                ("{synthesis_tool}", _tool),
                ("{process_constraints}", _pcon),
            ):
                system_prompt = system_prompt.replace(_ph, str(_val))

            # QSPI-slave frontend/IO-subsystem block: hand the RTL implementer the
            # COMPLETE chassis QSPI-slave protocol (cmd 0x02 write / 0x03 read +
            # dummy-byte turnaround / 0x05 read_status returning DONE+ERROR;
            # MSB-nibble-first; drive-on-falling / sample-on-rising; prefetch the
            # registered read DURING the dummy phase so read data is not launched a
            # nibble early). Injected only for the bus-boundary block.
            if _is_qspi_frontend_block(block_name, description):
                _fe_skill = _load_skills("qspi_slave_frontend_protocol")
                if _fe_skill:
                    system_prompt = (
                        system_prompt
                        + "\n\n# Reference Skill (QSPI-slave frontend protocol "
                        "completeness -- MANDATORY for this block)\n\n"
                        + _fe_skill
                    )

            # C12: snapshot the pre-call artifact. A PRE-EXISTING rtl file must
            # not masquerade as this call's output -- observed: two consecutive
            # "generation" calls logged success while the on-disk stub's
            # bytes/mtime never changed, so retries burned attempts against
            # RTL that was never regenerated.
            import hashlib as _hashlib
            import time as _time_mod

            rtl_path = Path(project_root) / rtl_target if project_root else Path(rtl_target)
            _pre_sha, _pre_mtime = "", 0
            if rtl_path.exists():
                try:
                    _pre_sha = _hashlib.sha1(rtl_path.read_bytes()).hexdigest()
                    _pre_mtime = rtl_path.stat().st_mtime_ns
                except OSError:
                    pass
            _call_start = _time_mod.time()

            run_name = f"Generate Verilog [{block_title}]{retry_label}"
            await self.llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name=run_name,
            )

            if rtl_path.exists():
                _now_sha, _now_mtime = "", 0
                try:
                    _now_sha = _hashlib.sha1(rtl_path.read_bytes()).hexdigest()
                    _now_mtime = rtl_path.stat().st_mtime_ns
                except OSError:
                    pass
                # Fresh iff the file appeared during this call, or its bytes
                # changed, or it was re-written in place (mtime advanced --
                # tolerates a genuinely identical re-emission).
                if (not _pre_sha or _now_sha != _pre_sha
                        or _now_mtime != _pre_mtime):
                    return {"rtl_path": str(rtl_path)}
                # unchanged pre-existing file -> NOT this call's output; try
                # to recover a genuinely-fresh per-call artifact instead.
            recovered = _recover_codex_artifact(
                project_root, rtl_target, min_mtime=_call_start)
            if recovered:
                rtl_path.parent.mkdir(parents=True, exist_ok=True)
                rtl_path.write_text(recovered, encoding="utf-8")
                return {"rtl_path": str(rtl_path)}
            return {"error": f"Agent did not write RTL to {rtl_target}"
                             + (" (pre-existing file unchanged -- a stale "
                                "artifact is not success)" if _pre_sha else "")}

    def _parse_response(
        self, content: str, block_name: str
    ) -> tuple[str, dict]:
        """Extract Verilog code and port info from LLM response."""
        import json
        import re

        # Extract Verilog code block
        verilog_match = re.search(
            r"```(?:verilog|v)?\s*\n(.*?)```",
            content,
            re.DOTALL,
        )
        if verilog_match:
            verilog = verilog_match.group(1).strip()
        else:
            # Assume the entire response is Verilog if no code block found
            verilog = content.strip()

        # Extract JSON port info
        ports = {}
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                port_info = json.loads(json_match.group(1))
                ports = port_info.get("ports", port_info)
            except json.JSONDecodeError:
                pass

        # Remove JSON block from verilog if it got included
        if "```json" in verilog:
            verilog = verilog[:verilog.index("```json")].strip()

        # Validate: reject error messages and prose written as Verilog
        if not verilog or "[ClaudeLLM error:" in verilog:
            raise ValueError(f"RTL generation returned error, not Verilog: {verilog[:200]}")
        first_nonblank = verilog.lstrip()[:10]
        if first_nonblank.startswith(("##", "# ", "---", "The ", "I ", "Right")):
            raise ValueError(f"RTL response contains prose, not Verilog: {verilog[:200]}")
        if not re.search(r"^\s*module\s+\w+", verilog, re.MULTILINE):
            raise ValueError(f"RTL response does not contain a Verilog module declaration: {verilog[:200]}")

        return verilog, ports
