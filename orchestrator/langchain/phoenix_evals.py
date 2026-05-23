# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation metrics for the coresmith ASIC pipeline.

Logs ASIC-specific evaluation metrics as OpenTelemetry span attributes
for analysis and filtering.

Metrics:
    - rtl_syntax_valid: Whether generated RTL passes Verilator lint
    - functional_match: Whether RTL simulation matches Python golden model
    - gate_count: Number of gates after Yosys synthesis
    - timing_met: Whether STA reports positive slack
    - iterations_to_converge: Number of generate/test attempts before success
"""

from __future__ import annotations

from opentelemetry import trace


def log_rtl_syntax_valid(valid: bool, block_name: str, errors: str = "") -> None:
    """Record whether generated RTL passed lint checks."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("eval.rtl_syntax_valid", valid)
        span.set_attribute("eval.block_name", block_name)
        if errors:
            span.set_attribute("eval.lint_errors", errors[:1000])


def log_functional_match(passed: bool, block_name: str, log: str = "") -> None:
    """Record whether RTL simulation matched the Python golden model."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("eval.functional_match", passed)
        span.set_attribute("eval.block_name", block_name)
        if log:
            span.set_attribute("eval.sim_log", log[:1000])


def log_gate_count(gate_count: int, block_name: str) -> None:
    """Record gate count from synthesis."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("eval.gate_count", gate_count)
        span.set_attribute("eval.block_name", block_name)


def log_timing_met(met: bool, block_name: str, worst_slack_ns: float | None = None) -> None:
    """Record whether STA timing constraints were met."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("eval.timing_met", met)
        span.set_attribute("eval.block_name", block_name)
        if worst_slack_ns is not None:
            span.set_attribute("eval.worst_slack_ns", worst_slack_ns)


def log_iterations_to_converge(iterations: int, block_name: str, success: bool) -> None:
    """Record how many attempts were needed to converge (or total if failed)."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("eval.iterations_to_converge", iterations)
        span.set_attribute("eval.converged", success)
        span.set_attribute("eval.block_name", block_name)
