# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TimingClosureAgent -- Analyzes OpenSTA timing reports and modifies RTL
to fix timing violations.

Strategies:
1. Pipeline insertion: add register stages to break long combinational paths
2. Logic restructuring: refactor wide muxes, deep logic cones
3. Clock constraint relaxation: suggest lower target frequency
4. Architecture escalation: signal that the block needs fundamental redesign

All modifications are traced via OpenTelemetry for auditability.
"""

from __future__ import annotations

from orchestrator._timeouts import scaled
import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .coresmith_llm import DEFAULT_MODEL, ClaudeLLM

_tracer = trace.get_tracer(__name__)

# Load system prompt from template file, fall back to inline
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "timing_closure.md"
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = """\
You are an expert ASIC timing closure engineer. You analyze static timing
analysis (STA) reports from OpenSTA and modify Verilog RTL to fix violations.

Given:
- STA report with critical path details
- Current Verilog RTL source
- Target clock frequency

Your strategies (in order of preference):
1. PIPELINE: Insert pipeline registers to break the critical path.
   - Identify the combinational path endpoints
   - Add a register stage in the middle
   - Update ready/valid handshaking for added latency
2. RESTRUCTURE: Refactor deep combinational logic.
   - Break wide multiplexers into tree structures
   - Pre-compute partial results in previous cycles
3. CONSTRAINT: Suggest relaxing the clock target if the violation is small.
4. ESCALATE: If the violation is fundamental (>50% of clock period),
   signal that the block needs architectural redesign.

Output format:
1. Modified Verilog source (complete module)
2. JSON block describing the changes:
   ```json
   {
     "strategy": "PIPELINE|RESTRUCTURE|CONSTRAINT|ESCALATE",
     "stages_added": 0,
     "latency_change": 0,
     "interface_changed": false,
     "description": "..."
   }
   ```

IMPORTANT: If you insert pipeline stages, update the module's latency
documentation and any ready/valid backpressure logic.
"""


class TimingClosureAgent:
    """Agent for timing closure optimization."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="CORESMITH_TIMING_CLOSURE_TIMEOUT"),
        )

    async def fix_timing(
        self,
        block_name: str,
        rtl_source: str,
        sta_report: str,
        target_clock_mhz: float,
        worst_slack_ns: float | None = None,
    ) -> dict[str, Any]:
        """
        Analyze timing violations and produce fixed RTL.

        Args:
            block_name: Block with timing violation
            rtl_source: Current Verilog source
            sta_report: OpenSTA report text
            target_clock_mhz: Target clock frequency
            worst_slack_ns: Worst negative slack

        Returns:
            Dict with: verilog, strategy, stages_added, latency_change,
                       interface_changed, escalate
        """
        block_title = block_name.replace("_", " ").title()
        span_name = f"Timing Closure [{block_title}]"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("target_clock_mhz", target_clock_mhz)
            if worst_slack_ns is not None:
                span.set_attribute("worst_slack_ns", worst_slack_ns)

            clock_period = 1000.0 / target_clock_mhz

            user_message = (
                f"Fix timing violations in '{block_name}'.\n\n"
                f"Target clock: {target_clock_mhz} MHz ({clock_period:.2f} ns period)\n"
                f"Worst slack: {worst_slack_ns} ns\n\n"
                f"--- STA Report ---\n{sta_report[-3000:]}\n\n"
                f"--- Current RTL ---\n```verilog\n{rtl_source}\n```"
            )

            try:
                content = await self.llm.call(
                    system=SYSTEM_PROMPT,
                    prompt=user_message,
                    run_name=f"Timing Closure [{block_title}]",
                )

                # Parse Verilog
                verilog_match = re.search(
                    r"```(?:verilog|v)?\s*\n(.*?)```", content, re.DOTALL
                )
                verilog = verilog_match.group(1).strip() if verilog_match else rtl_source

                # Parse JSON metadata
                json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
                metadata = {}
                if json_match:
                    try:
                        metadata = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass

                strategy = metadata.get("strategy", "UNKNOWN")

                return {
                    "verilog": verilog,
                    "strategy": strategy,
                    "stages_added": metadata.get("stages_added", 0),
                    "latency_change": metadata.get("latency_change", 0),
                    "interface_changed": metadata.get("interface_changed", False),
                    "escalate": strategy == "ESCALATE",
                    "description": metadata.get("description", ""),
                }

            except Exception as e:
                return {
                    "verilog": rtl_source,  # Return unchanged RTL
                    "strategy": "ERROR",
                    "stages_added": 0,
                    "latency_change": 0,
                    "interface_changed": False,
                    "escalate": True,
                    "description": f"TimingClosureAgent error: {e}",
                }
