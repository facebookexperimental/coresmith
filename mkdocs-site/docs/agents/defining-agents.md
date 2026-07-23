# How agents are defined

An "agent" in coresmith is a thin Python wrapper around an LLM call. There is **no LangChain runnable**, no agent framework, no tool registry. Every agent is just:

1. A class with an `__init__` that constructs a `ClaudeLLM`.
2. An `async def` entry point that builds a prompt, calls `self.llm.call(system=..., prompt=...)`, parses the response, and writes outputs to disk.

That's it. Agents live in [`orchestrator/langchain/agents/`](https://github.com/facebookexperimental/coresmith/tree/main/orchestrator/langchain/agents) (the `langchain` directory name is historical — there is no LangChain dependency at runtime).

## The agent shape

```python
# orchestrator/langchain/agents/uarch_spec_generator.py

class UarchSpecGenerator:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from orchestrator.langchain.agents.coresmith_llm import block_model
        model = model or block_model()
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="CORESMITH_UARCH_TIMEOUT"),
        )

    async def generate(self, *, block_name, python_source, description, ...):
        system_prompt = _load_prompt("uarch_spec_generator.md")
        user_message = _build_user_message(...)
        response_text = await self.llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=f"uarch_spec:{block_name}",
        )
        return _parse_response(response_text)
```

Every agent in the codebase follows this template. The interesting decisions are:

- **Which model to use** — Opus (`DEFAULT_MODEL`) for load-bearing steps (integration lead, integration review, contract audit, backend EDA) vs Sonnet (`BLOCK_MODEL`) for per-block work.
- **Which timeout** — each agent has its own `CORESMITH_<NAME>_TIMEOUT` env var.
- **Which prompt file** — every agent loads one `.md` from `orchestrator/langchain/prompts/`.
- **Whether to expose CLI tools** — by default the LLM has `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, etc. `BackendEDAAgent` disables them so the LLM only emits text.

## Where each agent is invoked from

| Agent | Invoked by graph node |
|---|---|
| `UarchSpecGenerator` | `pipeline_graph.generate_uarch_spec_node` |
| `RTLGeneratorAgent` | `pipeline_graph.generate_rtl_node` |
| `TestbenchGeneratorAgent` | `pipeline_graph.generate_testbench_node` |
| `DebugAgent` | `pipeline_graph.diagnose_node` |
| `IntegrationReviewAgent` | `pipeline_graph.integration_review_node` (per tier) |
| `IntegrationLeadAgent` | `pipeline_graph.integration_check_node` |
| `IntegrationTestbenchGenerator` | `pipeline_graph.integration_dv_node` |
| `ValidationDVGenerator` | `pipeline_graph.validation_dv_node` |
| `ContractAuditAgent` | `pipeline_graph._run_top_level_contract_audit` |
| `BackendEDAAgent` | every backend & tapeout EDA node |
| `TimingClosureAgent` | invoked from `diagnose_tapeout` and `timing_signoff` when violations are repairable |

## The disk-first contract

Agents **read context from disk** and **write outputs to disk**. State only carries paths. This has two consequences:

1. **You can drop into a run and edit any artifact.** The outer agent's `fix_rtl` / `fix_tb` actions exploit this — the agent edits `rtl/foo.v` in place and resumes the graph, which re-runs the failing stage without an LLM call.
2. **Agents don't pass blobs through the graph.** State stays small; serialization is fast; checkpoints are tiny.

What an agent typically reads from disk:

- The block's RTL (`rtl/<block>/<block>.v`)
- The testbench (`tb/cocotb/test_<block>.py`)
- The uArch spec (`arch/uarch_specs/<block>.md`)
- Accumulated constraints (`.coresmith/blocks/<block>/constraints.json`)
- Failure logs (`.coresmith/step_logs/<block>/<phase>_attempt_N.log`)
- VCD waveform (`sim_build/<block>/dump.vcd`)
- WaveKit audit (`sim_build/<block>/wavekit_audit.json`)
- Architecture docs (`arch/ers_spec.md`, `arch/frd_spec.md`, etc.)
- Block diagram (`.coresmith/block_diagram.json`)

What an agent writes:

- The artifact it is responsible for (RTL, TB, spec, diagnosis JSON).
- Optionally a constraints update (RTL/TB/Debug agents may append to `.coresmith/blocks/<block>/constraints.json`).

## Telemetry

Every `ClaudeLLM.call()` writes a JSONL record to `<project_root>/.coresmith/llm_calls.jsonl` with system prompt, user prompt, response, duration, token counts, cache hits, and cost. Every call also emits an OTel span (`orchestrator/langchain/agents/coresmith_llm.py:240-266`).

The circuit breaker (`coresmith_llm.py:45-108`) opens after 3 consecutive LLM failures and auto-resets after 60s of inactivity. The stall detector (`coresmith_llm.py:1136`) kills the CLI process if it produces no stdout/stderr for `_STALL_THRESHOLD_S` (1200s, scaled by `CORESMITH_TIMEOUT_MULTIPLIER`).

## Prompts as the public interface

Every agent's behavior is controlled by its prompt markdown file. Editing `orchestrator/langchain/prompts/rtl_generator.md` is the canonical way to change RTL output across the entire pipeline. The agents themselves are mostly glue:

- Load the prompt → build the user message → call the LLM → parse → write to disk.

See the [Prompt registry](prompts.md) for a list of every prompt and what it controls.
