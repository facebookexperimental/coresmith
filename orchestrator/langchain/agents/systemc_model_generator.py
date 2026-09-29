# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The SystemC block-model author (B2), called by the uArch phase after a
block's spec is approved. The agent writes ``model/<block>_model.cpp``
against the generated header; the caller verifies the file exists and
compiles, and re-invokes with the compiler log to repair (bounded)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from opentelemetry import trace

from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM, scaled
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

_tracer = trace.get_tracer(__name__)
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "systemc_model_generator.md"
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


class SystemCModelGenerator:
    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import block_model
        self.llm = ClaudeLLM(model=model or block_model(),
                             timeout=scaled(2400, env="CORESMITH_SYSTEMC_TIMEOUT"))

    async def generate(self, block_name: str, *, project_root: str, header_path: str,
                       compiler_log: str = "", attempt: int = 1) -> dict:
        block_title = block_name.replace("_", " ").title()
        parts = [f"Implement the SystemC model of block '{block_name}'.", "",
                 "## Working files (read them)",
                 f"- Generated header (bound by name, keep intact): {header_path}",
                 "- Common header: model/cs_model_common.h",
                 f"- uArch spec: arch/uarch_specs/{block_name}.md",
                 f"- Contract slice: .coresmith/blocks/{block_name}/contract_slice.json",
                 "", "## Output", f"Write model/{block_name}_model.cpp and reply with the JSON block."]
        if compiler_log:
            parts += ["", f"## Compiler errors from attempt {attempt - 1} (fix them in place)",
                      "```", compiler_log[-6000:], "```"]
        from orchestrator.state_store.rulings import rulings_section
        parts.append(rulings_section(project_root, consumer="systemc_model", block=block_name))
        with _tracer.start_as_current_span(f"SystemC Model [{block_title}]") as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("attempt", attempt)
            content = await self.llm.call(system=SYSTEM_PROMPT, prompt="\n".join(parts),
                                          run_name=f"SystemC Model [{block_title}]"
                                          + (f" (repair {attempt})" if compiler_log else ""))
        out = _parse(content)
        out["cpp_path"] = str(Path(project_root) / "model" / f"{block_name}_model.cpp")
        out["written"] = Path(out["cpp_path"]).exists()
        return out
