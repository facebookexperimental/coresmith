# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Block Diagram system prompt goes through ``str.format`` (block_diagram.py);
a literal ``{...}`` in the prompt text raised IndexError on every run of the
SoC-stages branch (found on the first benchmark run). Guard the fields."""
import string

from orchestrator.architecture.specialists import block_diagram as bd

_ALLOWED = {"benchmark_context", "constraint_context", "feedback_context"}


def test_block_diagram_prompt_has_only_known_format_fields():
    fields = {f for _, f, _, _ in string.Formatter().parse(bd.SYSTEM_PROMPT) if f is not None}
    assert fields <= _ALLOWED, fields - _ALLOWED
    assert "" not in fields, "positional {} / {...} in the prompt text"


def test_block_diagram_prompt_formats():
    out = bd.SYSTEM_PROMPT.format(benchmark_context="", constraint_context="", feedback_context="")
    assert '"fabric": {...}' in out
