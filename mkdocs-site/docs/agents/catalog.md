# Agent catalog

A per-agent reference. Each entry covers: where it lives, what node invokes it, what model/prompt it uses, what it reads from disk, what it writes, and any non-obvious handling.

## UarchSpecGenerator

- **File:** `orchestrator/langchain/agents/uarch_spec_generator.py`
- **Entry:** `async def generate(*, block_name, python_source, description, feedback, previous_spec, constraints, project_root)`
- **Model / timeout:** `BLOCK_MODEL` (Sonnet) / `CORESMITH_UARCH_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/uarch_spec_generator.md`
- **Invoked by:** `pipeline_graph.generate_uarch_spec_node:398`

Writes a markdown microarchitecture spec for a block (datapath, FSM, interfaces, timing). Reads `arch/ers_spec.md`, `.coresmith/block_diagram.json`, `arch/frd_spec.md` for context. In revision mode (when `previous_spec` is provided with `feedback`), instructs the LLM to amend the old spec rather than start over.

Returns `{"spec_text": str, "spec_summary": dict, "block_name": str}`.

## RTLGeneratorAgent

- **File:** `orchestrator/langchain/agents/rtl_generator.py`
- **Entry:** `async def generate(*, block_name, description, attempt, rtl_target, python_source_path, project_root, ...)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_RTL_TIMEOUT` (default 1800s)
- **Prompt:** `prompts/rtl_generator.md` (with a strong inline fallback covering Verilog-2005, AXI-Stream, synchronous reset, fixed-point, Sky130 rules)
- **Invoked by:** `pipeline_graph.generate_rtl_node:488`

Generates synthesizable Verilog from the uArch spec and Python golden model. Reads the spec, ERS, accumulated constraints, the golden model, the block diagram, and (on retry) `previous_error.txt`.

Has a **Codex artifact recovery** path (`rtl_generator.py:104-124`): if Codex writes the file directly but returns a placeholder, the agent walks recent `codex-call-*` directories to find the real file on disk. Validates that the response contains a `module` declaration before accepting it.

On retry (`attempt > 1`), the prompt instructs the LLM to use `Edit` for surgical fixes before considering a full rewrite.

## TestbenchGeneratorAgent

- **File:** `orchestrator/langchain/agents/testbench_generator.py`
- **Entry:** `async def generate(*, block_name, rtl_path, python_source_path, testbench_path, project_root)`
- **Model / timeout:** `BLOCK_MODEL` (Sonnet) / `CORESMITH_TB_TIMEOUT` (default 1800s)
- **Prompt:** `prompts/testbench_generator.md` (mandatory; raises if missing)
- **Invoked by:** `pipeline_graph.generate_testbench_node:706`

Writes a cocotb testbench that exercises the RTL against the Python golden model. Reads the actual RTL file (to extract exact port names, line 114), the golden model, the uArch spec, constraints, and `arch/DV_RULES.md` if present.

**Post-validation** (`testbench_generator.py:148-165`): the agent counts `@cocotb.test()` functions in the output and raises `RuntimeError` if zero — this catches LLMs that produce a Python file with no actual tests.

## IntegrationLeadAgent

- **File:** `orchestrator/langchain/agents/integration_lead.py`
- **Entry:** `async def integrate(*, design_name, block_rtl_sources, block_port_summaries, connections, prd_summary, output_path)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_INTEGRATION_LEAD_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/integration_lead.md`
- **Invoked by:** `pipeline_graph.integration_check_node:2139`

Generates the top-level `chip_top.v` wiring all block RTL together. Reports any port-width / direction / protocol mismatches it finds.

**The famous JSON-vs-disk bug** (`integration_lead.py:95-162`): Codex sometimes writes the full top to disk via its `Write` tool and then returns `{"verilog": "\`include \"<output_path>\""}` as the JSON response. That placeholder is a self-include and would corrupt the file if blindly written. `_has_real_module()` checks both the LLM response and the on-disk file, picks the one with a real `module ` declaration, and raises if *neither* is valid (so the graph retries).

Returns `{"rtl_path", "mismatches", "module_name", "wire_count", "skipped_connections", "notes"}`.

## IntegrationReviewAgent

- **File:** `orchestrator/langchain/agents/integration_review_agent.py`
- **Entry:** `async def review(*, block_names, project_root)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_INTEGRATION_REVIEW_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/integration_review.md`
- **Invoked by:** `pipeline_graph.integration_review_node:1784` (once per tier)

Reads every uArch spec in the current tier and the architecture connections, then **edits the specs on disk** to fix mismatches. Returns `{summary, issues_found, issues_fixed}`.

**Non-obvious behavior:** the agent always finds something to edit — `issues_fixed > 0` is the steady state, not an error. The orchestrator's default action is `approve` regardless. Set `CORESMITH_STRICT_INTEGRATION_REVIEW=1` to auto-`revise` on `issues_fixed > 0` (old behavior; useful when you suspect spec drift).

Per-tier filtering (lines 54–85): cross-tier and future-tier connections are excluded from the review scope so the agent doesn't flag deferred work as a current-tier issue.

## IntegrationTestbenchGenerator

- **File:** `orchestrator/langchain/agents/integration_testbench_generator.py`
- **Entry:** `async def generate(*, design_name, top_rtl_source, block_summaries, connections, prd_summary, block_rtl_paths, output_path, prior_failure)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_INTEGRATION_TB_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/integration_testbench.md`
- **Invoked by:** `pipeline_graph.integration_dv_node:2589`

Generates a chip-level cocotb testbench: reset, smoke handshakes, throughput, backpressure. Self-contained — there's no Python golden model at the chip level, so the agent must derive expected behavior from the spec & connections.

Has a disk-fallback (`lines 154-161`): if the agent doesn't write a testbench but a file with `@cocotb.test()` already exists at `output_path`, the agent uses it.

## ValidationDVGenerator

- **File:** `orchestrator/langchain/agents/validation_dv_generator.py`
- **Entry:** `async def generate(*, design_name, top_rtl_path, top_rtl_source, block_summaries, connections, ers_context, block_rtl_paths, output_path, prior_failure)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_VALIDATION_DV_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/validation_dv.md`
- **Invoked by:** `pipeline_graph.validation_dv_node:3084`

Generates KPI/application-level cocotb tests driven by the ERS. The generated TB **must** define a `REQUIREMENT_COVERAGE` dict; the agent rejects testbenches without it.

Has a sanitization post-pass (`lines 164-187`) that fixes known LLM-introduced bugs in wide-bus reads and backpressure gap patterns.

## DebugAgent

- **File:** `orchestrator/langchain/agents/debug_agent.py`
- **Entry:** `async def analyze(*, block_name, phase, project_root, mode)`
- **Model / timeout:** `BLOCK_MODEL` (Sonnet) / 900s
- **Prompt:** `prompts/debug_agent.md` (with a strong inline fallback for lint / sim / synth diagnoses)
- **Invoked by:** `pipeline_graph.diagnose_node:1021`

Disk-first: reads the failure log, VCD, WaveKit audit, RTL, TB, uArch spec, accumulated constraints, attempt history, block diagram, and ERS. Writes the diagnosis to `.coresmith/blocks/<block>/diagnosis.json`.

Two modes:

- `"debug"` (default) — diagnose this specific failure.
- `"architecture_review"` — escalation; called after repeated failures to decide whether the fix is local or requires an architecture change.

Diagnosis categories: `LOGIC_ERROR`, `TIMING_ISSUE`, `INTERFACE_MISMATCH`, `RESET_BUG`, `ARITHMETIC_ERROR`, `STATE_MACHINE_BUG`.

The agent can append new constraints to `.coresmith/blocks/<block>/constraints.json` so the next RTL/TB generation avoids the same bug.

## ContractAuditAgent

- **File:** `orchestrator/langchain/agents/contract_audit_agent.py`
- **Entry:** `async def analyze(*, stage, project_root, context_path, output_path)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / 900s
- **Prompt:** `prompts/contract_audit.md`
- **Invoked by:** `pipeline_graph._run_top_level_contract_audit:2984`

Runs after an integration DV or validation DV failure. Produces a structured JSON diagnosis that includes:

```python
{
  "stage": "integration_dv" | "validation_dv",
  "passed": False,
  "category": "UARCH_INTERFACE_CONTRACT_ERROR" | "RTL_BUG" | "TESTBENCH_BUG" | ...,
  "contract_failure": bool,                   # True if uArch/interface issue
  "missing_or_broken_contract": str,
  "first_divergence": {
    "summary": str,
    "golden_observation": str,
    "rtl_observation": str,
    "vcd_signals": list[str],
    "log_refs": list[str],
  },
  "local_fix_possible": bool,
  "recommended_action": "fix_rtl" | "fix_tb" | "revise_uarch" | "ask_human",
  "suggested_fix": str,
  "required_uarch_patch": {"rationale": str, "sections_to_replace": list},
  "affected_blocks": list[str],
  "confidence": float,
  "outer_agent_summary": str,
}
```

Writes to `.coresmith/contract_audit/<stage>_contract_audit.json`. Normalization rules ensure `contract_failure=True` and `recommended_action="revise_uarch"` whenever the category indicates an interface/architecture issue.

**Reliability note:** per CLAUDE.md, the contract audit is more precise than its `confidence` field suggests. If `local_fix_possible: true` with concrete `suggested_fix`, the fix is usually small.

## BackendEDAAgent

- **File:** `orchestrator/langchain/agents/backend_eda_agent.py`
- **Entry:** two:
    - `async def adapt_script(*, baseline_script, context)` — adapt a template TCL/shell script
    - `async def analyze(*, context)` — analyze a tool's output
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / 180s default
- **Prompt:** one per step — `prompts/backend_{synthesis,pnr,drc,lvs,timing_signoff,mpw_precheck}.md` (the `_llm` variants are for analysis turns)
- **Invoked by:** every node in `backend_graph.py` and `tapeout_graph.py` that runs an EDA tool

Built with `disable_tools=True` (line 103) — no shell, no file edits. The LLM gets text in, returns text out. The graph code runs the actual tools and feeds the parsed output back.

Falls back to the unmodified baseline script on LLM failure rather than raising — this keeps the graph moving even if the LLM hiccups.

## TimingClosureAgent

- **File:** `orchestrator/langchain/agents/timing_closure.py`
- **Entry:** `async def fix_timing(*, block_name, rtl_source, sta_report, target_clock_mhz, worst_slack_ns)`
- **Model / timeout:** `DEFAULT_MODEL` (Opus 4.8) / `CORESMITH_TIMING_CLOSURE_TIMEOUT` (default 2700s)
- **Prompt:** `prompts/timing_closure.md`
- **Invoked by:** backend / tapeout diagnosis paths when STA reports a violation that looks repairable

Returns `{"verilog", "strategy", "stages_added", "latency_change", "interface_changed", "escalate", "description"}`. Strategy is one of `PIPELINE` (insert flops), `RESTRUCTURE` (refactor logic), `CONSTRAINT` (relax target), `ESCALATE` (needs architecture change).

## OpencodePatch

- **File:** `orchestrator/langchain/agents/opencode_patch.py`
- **Purpose:** Monkey-patch for using the local `opencode` CLI (Qwen / local model) instead of Claude or Codex. Not a full agent — a drop-in replacement for `ClaudeLLM._generate_via_cli`.

Useful for cost-sensitive runs or local-model evaluation. Not the production path.
