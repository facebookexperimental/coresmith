# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for constraint accumulation logic:
- RTLConstraint / BlockQuestion dataclasses
- _infer_scope helper
- Constraint deduplication logic
- Port-signature extraction and comparison
"""

import hashlib
import re
from dataclasses import dataclass, field

import pytest

# ---------------------------------------------------------------------------
# Dataclass definitions (previously in orchestrator.temporal)
# ---------------------------------------------------------------------------


@dataclass
class RTLConstraint:
    """A constraint learned from debug analysis or human input."""
    rule: str
    scope: str
    source: str
    attempt: int

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "scope": self.scope,
            "source": self.source,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RTLConstraint":
        return cls(
            rule=d["rule"],
            scope=d["scope"],
            source=d["source"],
            attempt=d["attempt"],
        )

    def __eq__(self, other):
        if not isinstance(other, RTLConstraint):
            return NotImplemented
        return (self.rule == other.rule and self.scope == other.scope
                and self.source == other.source and self.attempt == other.attempt)


@dataclass
class BlockQuestion:
    """A question surfaced to the human during pipeline execution."""
    block_name: str
    question: str
    category: str
    attempt: int
    choices: list = field(default_factory=lambda: ["skip", "escalate"])


def _infer_scope(category: str) -> str:
    """Map a DebugAgent failure category to a constraint scope."""
    return {
        "LOGIC_ERROR": "rtl",
        "TIMING_ISSUE": "timing",
        "INTERFACE_MISMATCH": "interface",
        "RESET_BUG": "rtl",
        "ARITHMETIC_ERROR": "rtl",
        "STATE_MACHINE_BUG": "rtl",
        "AGENT_ERROR": "rtl",
    }.get(category, "rtl")


def _extract_port_signature(verilog_source: str) -> str:
    """Extract a canonical hash of the module port declarations.

    Parses input/output/inout port declarations, normalizes whitespace,
    sorts them, and returns a SHA-256 hex digest. Internal logic is ignored.
    """
    port_pattern = re.compile(
        r'^\s*(input|output|inout)\s+(wire|reg)?\s*(\[.*?\])?\s*(\w+)',
        re.MULTILINE,
    )
    ports = []
    for m in port_pattern.finditer(verilog_source):
        direction = m.group(1).strip()
        width = (m.group(3) or "").replace(" ", "")
        name = m.group(4).strip()
        ports.append(f"{direction} {width} {name}")
    ports.sort()
    canonical = "\n".join(ports)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# RTLConstraint dataclass
# ---------------------------------------------------------------------------


class TestRTLConstraint:
    def test_to_dict(self):
        c = RTLConstraint(
            rule="MUST register m_tdata output",
            scope="rtl",
            source="debug_agent",
            attempt=2,
        )
        d = c.to_dict()
        assert d == {
            "rule": "MUST register m_tdata output",
            "scope": "rtl",
            "source": "debug_agent",
            "attempt": 2,
        }

    def test_from_dict(self):
        d = {
            "rule": "MUST NOT use blocking assigns",
            "scope": "rtl",
            "source": "human",
            "attempt": 3,
        }
        c = RTLConstraint.from_dict(d)
        assert c.rule == "MUST NOT use blocking assigns"
        assert c.scope == "rtl"
        assert c.source == "human"
        assert c.attempt == 3

    def test_roundtrip(self):
        c = RTLConstraint(
            rule="MUST add pipeline register",
            scope="timing",
            source="sta",
            attempt=1,
        )
        assert RTLConstraint.from_dict(c.to_dict()) == c


# ---------------------------------------------------------------------------
# BlockQuestion dataclass
# ---------------------------------------------------------------------------


class TestBlockQuestion:
    def test_defaults(self):
        q = BlockQuestion(
            block_name="scrambler",
            question="Is the LFSR polynomial correct?",
            category="LOGIC_ERROR",
            attempt=2,
        )
        assert q.choices == ["skip", "escalate"]
        assert q.block_name == "scrambler"

    def test_custom_choices(self):
        q = BlockQuestion(
            block_name="fft",
            question="Which butterfly structure?",
            category="LOGIC_ERROR",
            attempt=1,
            choices=["radix-2", "radix-4", "escalate"],
        )
        assert len(q.choices) == 3


# ---------------------------------------------------------------------------
# _infer_scope helper
# ---------------------------------------------------------------------------


class TestInferScope:
    @pytest.mark.parametrize(
        "category,expected_scope",
        [
            ("LOGIC_ERROR", "rtl"),
            ("TIMING_ISSUE", "timing"),
            ("INTERFACE_MISMATCH", "interface"),
            ("RESET_BUG", "rtl"),
            ("ARITHMETIC_ERROR", "rtl"),
            ("STATE_MACHINE_BUG", "rtl"),
            ("AGENT_ERROR", "rtl"),
            ("UNKNOWN_CATEGORY", "rtl"),  # fallback
        ],
    )
    def test_mapping(self, category, expected_scope):
        assert _infer_scope(category) == expected_scope


# ---------------------------------------------------------------------------
# Constraint deduplication (unit-level logic test)
# ---------------------------------------------------------------------------


class TestConstraintDeduplication:
    def test_deduplicate_by_rule_text(self):
        """Simulate the dedup logic from BlockRTLWorkflow.run()."""
        constraints: list[dict] = [
            {"rule": "MUST register output", "scope": "rtl", "source": "debug_agent", "attempt": 1},
        ]
        new_rules = ["MUST register output", "MUST NOT use latches"]

        existing_rules = {c["rule"] for c in constraints}
        for rule_text in new_rules:
            if rule_text not in existing_rules:
                constraints.append({
                    "rule": rule_text,
                    "scope": "rtl",
                    "source": "debug_agent",
                    "attempt": 2,
                })
                existing_rules.add(rule_text)

        assert len(constraints) == 2
        rules = [c["rule"] for c in constraints]
        assert "MUST register output" in rules
        assert "MUST NOT use latches" in rules


# ---------------------------------------------------------------------------
# Port-signature extraction
# ---------------------------------------------------------------------------


class TestPortSignature:
    SIMPLE_MODULE = """\
module scrambler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  s_tdata,
    input  wire        s_tvalid,
    output wire        s_tready,
    output wire [7:0]  m_tdata,
    output wire        m_tvalid,
    input  wire        m_tready
);
    // internal logic...
    reg [15:0] lfsr;
    assign m_tdata = s_tdata ^ lfsr[7:0];
endmodule
"""

    SAME_PORTS_DIFFERENT_LOGIC = """\
module scrambler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  s_tdata,
    input  wire        s_tvalid,
    output wire        s_tready,
    output wire [7:0]  m_tdata,
    output wire        m_tvalid,
    input  wire        m_tready
);
    // COMPLETELY different internal logic
    reg [31:0] lfsr;
    reg [7:0] buffer;
    assign m_tdata = buffer;
endmodule
"""

    DIFFERENT_PORTS = """\
module scrambler (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [15:0]  s_tdata,
    input  wire         s_tvalid,
    output wire         s_tready,
    output wire [15:0]  m_tdata,
    output wire         m_tvalid,
    input  wire         m_tready
);
    assign m_tdata = s_tdata;
endmodule
"""

    def test_same_ports_same_signature(self):
        sig1 = _extract_port_signature(self.SIMPLE_MODULE)
        sig2 = _extract_port_signature(self.SAME_PORTS_DIFFERENT_LOGIC)
        assert sig1 == sig2

    def test_different_ports_different_signature(self):
        sig1 = _extract_port_signature(self.SIMPLE_MODULE)
        sig2 = _extract_port_signature(self.DIFFERENT_PORTS)
        assert sig1 != sig2

    def test_signature_is_hex_string(self):
        sig = _extract_port_signature(self.SIMPLE_MODULE)
        assert len(sig) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in sig)

    def test_whitespace_insensitive(self):
        """Extra whitespace in port declarations should not change the sig."""
        module_extra_ws = """\
module scrambler (
    input   wire         clk,
    input   wire         rst_n,
    input   wire  [7:0]  s_tdata,
    input   wire         s_tvalid,
    output  wire         s_tready,
    output  wire  [7:0]  m_tdata,
    output  wire         m_tvalid,
    input   wire         m_tready
);
endmodule
"""
        sig1 = _extract_port_signature(self.SIMPLE_MODULE)
        sig2 = _extract_port_signature(module_extra_ws)
        assert sig1 == sig2

    def test_empty_module(self):
        sig = _extract_port_signature("")
        assert len(sig) == 64  # Still produces a valid hash (of empty string)
