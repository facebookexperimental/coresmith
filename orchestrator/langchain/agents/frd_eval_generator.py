# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The FRD-evaluation harness author (B2). After the SoC model builds and
smokes, the agent writes ``model/frd_eval/*.cpp`` -- a SystemC program that
drives ``soc_model_top`` through the mission stimulus and prints one
``FRD_EVAL {...}`` verdict per FRD requirement. The caller builds it, runs it
and re-invokes with the compiler/run log to repair (bounded)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from opentelemetry import trace

from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM, scaled
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

_tracer = trace.get_tracer(__name__)
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "frd_eval_generator.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8") if _PROMPT_FILE.exists() else ""
_SKILLS_TEXT = _load_skills("systemc_tlm_lt")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n# Reference Skill\n\n" + _SKILLS_TEXT


def _parse(content: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", content or "", re.S)
    if not m:
        return {"files_written": [], "notes": ""}
    try:
        return json.loads(m.group(1))
    except ValueError:
        return {"files_written": [], "notes": ""}


class FRDEvalGenerator:
    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import block_model
        self.llm = ClaudeLLM(model=model or block_model(),
                             timeout=scaled(3600, env="CORESMITH_FRD_EVAL_TIMEOUT"))

    async def generate(self, *, project_root: str, blocks: list[str], attempt: int = 1,
                       compiler_log: str = "", run_log: str = "", summary: dict | None = None,
                       arch: bool = False) -> dict:
        if arch:
            parts = ["Write the FRD evaluation harness for this SoC's EXECUTABLE ARCHITECTURE MODEL (the abstract "
                     "SystemC performance model, before decomposition).", "",
                     "## Working files (read them)",
                     "- Requirements to answer: model/arch/frd_eval/requirements.json (every id needs a verdict line)",
                     "- FRD: arch/frd_spec.md (acceptance criteria, the Mission-Scale Acceptance Test, model-check notes)",
                     "- Architecture spec: model/arch/arch_model.json (components, windows, latencies, links)",
                     "- Model top: model/arch/arch_model_top.h -- `arch_model_top top`; initiators `top.u_<name>[i]` are "
                     "`cs_arch_initiator` (set `->body = [&](cs_arch_initiator& me){...}` BEFORE sc_start; inside use "
                     "me.issue/rd64/wr64/compute(cycles)/idle(cycles)); targets `top.u_<name>` (`load_bin`, `mem`); "
                     "`top.stats_json(os, cycles)` gives per-link bytes/cycle, utilization, outstanding, energy",
                     "- Primitives: model/arch/cs_arch_common.h",
                     "- Workload data: inputs/acceptance_stimulus.py, inputs/references/ (measured triangle counts, oracle logs, "
                     "frame sizes) -- derive the transaction mix from MEASURED numbers, cite them in the evidence",
                     "", "## What the scenario must do",
                     "Reproduce the mission's traffic at the architecture level: per frame/iteration the bytes each initiator "
                     "moves to each target, the compute cycles it spends, its dependencies (e.g. GPU waits for the CPU's "
                     "submission; video DMA reads a full frame per vsync). Run enough iterations for steady state (or the "
                     "whole mission if it fits the time budget) and judge every PERF/IFACE/INV requirement the FRD's "
                     "'Model check' lines make observable here; cycle-based PERF verdicts use the model's cycle "
                     "accounting and say so. Print the stats JSON in the evidence of the requirements that depend on it.",
                     "", "## Output",
                     "Write model/arch/frd_eval/frd_eval.cpp (plus helpers under model/arch/frd_eval/) and reply with the JSON block.",
                     "Build: `make -C model/arch frd_eval/frd_eval`; run: `model/arch/frd_eval/frd_eval` (no arguments)."]
        else:
            parts = ["Write the FRD evaluation harness for this SoC's SystemC model.", "",
                     "## Working files (read them)",
                     "- Requirements to answer: model/frd_eval/requirements.json (every id needs a verdict line)",
                     "- FRD: arch/frd_spec.md (acceptance criteria, the Mission-Scale Acceptance Test, model-check notes)",
                     "- SoC model: model/soc_model_top.h (public u_<block> members; bind clk/rst_n; reset_all/dump_all)",
                     "- Block model headers: model/<block>_model.h for: " + ", ".join(blocks),
                     "- uArch specs: arch/uarch_specs/<block>.md (register maps, behaviour the model implements)",
                     "- Contract slices: .coresmith/blocks/<block>/contract_slice.json",
                     "- Mission stimulus: inputs/acceptance_stimulus.py and inputs/references/ (firmware, oracles, hashes)",
                     "- Common header: model/cs_model_common.h (cs_transact, cs_mem::load_bin, cs_clock_period)",
                     "", "## Output",
                     "Write model/frd_eval/frd_eval.cpp (plus any frd_eval/*.h|*.cpp helpers you need) and reply with the JSON block.",
                     "Build: `make -C model frd_eval/frd_eval`; run: `model/frd_eval/frd_eval` (no arguments)."]
        if compiler_log:
            parts += ["", f"## Compiler errors from attempt {attempt - 1} (fix them in place)",
                      "```", compiler_log[-8000:], "```"]
        if run_log:
            parts += ["", f"## Run log from attempt {attempt - 1}",
                      "The harness must print FRD_EVAL_DONE and exit 0 even when requirements fail; "
                      "a crash, hang or missing DONE is a harness defect -- fix it. A `fail` verdict that "
                      "is caused by a wrong check (not by the model) is also yours to fix; a genuine model "
                      "failure stays `fail` with the evidence.",
                      "```", run_log[-8000:], "```"]
        if summary:
            parts += ["", "## Verdict summary of the previous run", "```json", json.dumps(summary, indent=1)[:3000], "```"]
        from orchestrator.state_store.rulings import rulings_section
        parts.append(rulings_section(project_root, consumer="frd_eval"))
        with _tracer.start_as_current_span("FRD Evaluation Harness") as span:
            span.set_attribute("attempt", attempt)
            content = await self.llm.call(system=SYSTEM_PROMPT, prompt="\n".join(parts),
                                          run_name=("Architecture " if arch else "") + "FRD Evaluation Harness"
                                          + (f" (repair {attempt})" if attempt > 1 else ""))
        out = _parse(content)
        srcs = sorted((Path(project_root) / "model" / ("arch/frd_eval" if arch else "frd_eval")).glob("*.cpp"))
        out["sources"] = [str(s) for s in srcs]
        out["written"] = bool(srcs)
        return out
