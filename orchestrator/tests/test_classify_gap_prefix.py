# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the composition-gate prefix classifier.

The length-only heuristic classified the Opus codec's entropy coding bug (byte-exact
through byte 31, diverges at byte 32, 238B golden vs 50B composed) as a
'contract' gap -> full 8-block re-fan (twice, >55% of the run's tokens). The
prefix probe classifies framing-byte-exact-then-diverges as 'block_math' (a
single-block re-spec), and keeps a true offset-0 mismatch as 'contract'.
"""
from __future__ import annotations

from orchestrator.architecture.model_integration import (
    _classify_gap,
    first_divergence_offset,
)


def test_offset_basic():
    assert first_divergence_offset(b"abcdef", b"abcXef") == 3
    assert first_divergence_offset(b"abc", b"abc") == -1
    # composed is a clean prefix but shorter -> diverges at the shorter length
    assert first_divergence_offset(b"abcdef", b"abc") == 3


def test_framing_exact_then_diverges_is_block_math():
    golden = bytes(range(32)) + b"\x1b\x86\xc0" + bytes(200)   # 235 B
    composed = bytes(range(32)) + b"\x19\x3c\x38"              # 35 B, diverges @32
    assert _classify_gap("/x", golden, composed) == "block_math"


def test_divergence_at_offset_zero_is_contract():
    # no shared framing at all -> genuine width/framing contract gap
    assert _classify_gap("/x", b"\xaa" * 40, b"\xbb" * 12) == "contract"


def test_same_bytes_is_block_math_default():
    # equal-length value divergence -> block_math (unchanged behavior)
    assert _classify_gap("/x", b"\x01\x02\x03\x04", b"\x01\x02\xff\x04") == "block_math"


def test_prefix_probe_disabled_falls_back_to_length(monkeypatch):
    monkeypatch.setenv("CORESMITH_GATE_PREFIX_CLASSIFY", "0")
    golden = bytes(range(40))
    composed = bytes(range(20))   # prefix-exact but shorter
    # with the probe OFF, the length heuristic calls it contract
    assert _classify_gap("/x", golden, composed) == "contract"


def test_non_byteseq_unaffected():
    # dict outputs keep the shape heuristic (different keys -> contract)
    assert _classify_gap("/x", {"a": 1, "b": 2}, {"a": 1}) == "contract"
