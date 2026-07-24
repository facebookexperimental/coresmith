# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C16: the block-complexity gate must score a block's OWN python_source slice
(the architecture-assigned golden mapping), not the codec-specific BLOCK_ENTRY_HINTS
resolver -- which returned an empty slice (score 0, never flagged) for every
non-the video codec design. Reproduces the residual_recon_engine miss: a 6-algorithm /
582-LOC / cyclomatic-158 fusion that the legacy gate scored as 0."""

from orchestrator.langgraph import block_complexity as bc

# A tiny golden: two fat compute functions fusing several algorithm families
# (branchy, table-driven, transform-ish) + one small store helper.
_GOLDEN = '''
OC_TOKEN_MAP = [0] * 32
DEQUANT = [1] * 64

def _dc_unpredict(frags, refs):
    for i in range(len(frags)):
        if i > 0 and refs[i]:
            frags[i] += (frags[i-1] * 3 + 2) >> 2
        elif frags[i] & 1:
            frags[i] -= 1
        else:
            frags[i] ^= 0x7f
    return frags

def idct8x8(block):
    out = [0] * 64
    for y in range(8):
        for x in range(8):
            acc = 0
            for k in range(8):
                acc += block[y*8+k] * DEQUANT[k*8+x]
            out[y*8+x] = (acc + 8) >> 4
    return out

def token_decode(bits):
    v = 0
    while bits:
        b = bits.pop()
        if b == 0:
            v = (v << 1)
        elif b == 1:
            v = (v << 1) | 1
        else:
            break
    return OC_TOKEN_MAP[v & 31]

def store_put(mem, addr, val):
    mem[addr] = val & 0xffff
    return mem
'''


def _stats():
    return bc._parse_functions(_GOLDEN)


class TestPythonSourceSliceFns:
    def test_extracts_named_functions(self):
        fns = bc.python_source_slice_fns(
            "g.py:_dc_unpredict,idct8x8,token_decode", _stats())
        assert fns == ["_dc_unpredict", "idct8x8", "token_decode"]

    def test_bare_path_is_empty(self):
        assert bc.python_source_slice_fns("g.py", _stats()) == []
        assert bc.python_source_slice_fns("", _stats()) == []

    def test_unknown_names_dropped(self):
        assert bc.python_source_slice_fns("g.py:nope,idct8x8", _stats()) \
            == ["idct8x8"]


class TestComplexityGateScoresPythonSource:
    def test_fat_fusion_flagged_via_python_source(self):
        # The three compute functions together fuse several algorithms and are
        # branchy -- the gate must flag them when handed the slice, where the
        # legacy-hint resolver would score 0 (no hint / no name-substring match).
        stats = _stats()
        slice_fns = bc.python_source_slice_fns(
            "g.py:_dc_unpredict,idct8x8,token_decode", stats)
        est = bc.estimate_block_complexity(
            "residual_recon_engine", "g.py", stats=stats, slice_fns=slice_fns)
        legacy = bc.estimate_block_complexity(
            "residual_recon_engine", "g.py", stats=stats)  # hint path
        assert legacy["over_budget"] is False   # the miss: scored 0
        assert est["modeling_complexity"] > legacy["modeling_complexity"]

    def test_hint_path_empty_slice_scores_zero(self):
        # No BLOCK_ENTRY_HINTS + no name-substring match -> empty slice -> 0.
        est = bc.estimate_block_complexity(
            "residual_recon_engine", "g.py", stats=_stats())
        assert est["modeling_complexity"] == 0
        assert est["over_budget"] is False


class TestComplexityGateNotBypassedByQuestions:
    """C17: the complexity/decomposition gate must run whether or not the block
    diagram asked a clarifying question. Previously the post-question 'continue'
    route hard-wired Interface Definition, so any design whose diagram asked a
    question skipped the gate (residual_recon_engine was never checked)."""

    def test_clean_diagram_routes_to_complexity_gate(self, monkeypatch):
        from orchestrator.langgraph import architecture_graph as ag
        monkeypatch.delenv("CORESMITH_COMPLEXITY_GATE", raising=False)
        state = {"block_diagram": {"blocks": [{"name": "a"}], "questions": []}}
        assert ag.review_diagram(state) == "Complexity Review"

    def test_questioned_then_continue_STILL_runs_gate(self, monkeypatch):
        from orchestrator.langgraph import architecture_graph as ag
        monkeypatch.delenv("CORESMITH_COMPLEXITY_GATE", raising=False)
        # the C17 regression: 'continue' after a diagram question must NOT skip
        # straight to Interface Definition.
        for action in ("continue", "accept", "ok"):
            state = {"human_response": {"action": action}}
            assert ag.route_after_diagram_escalation(state) == "Complexity Review"

    def test_feedback_and_abort_unchanged(self):
        from orchestrator.langgraph import architecture_graph as ag
        assert ag.route_after_diagram_escalation(
            {"human_response": {"action": "feedback"}}) == "Block Diagram"
        assert ag.route_after_diagram_escalation(
            {"human_response": {"action": "abort"}}) == "Abort"

    def test_gate_disabled_falls_through(self, monkeypatch):
        from orchestrator.langgraph import architecture_graph as ag
        monkeypatch.setenv("CORESMITH_COMPLEXITY_GATE", "0")
        monkeypatch.setenv("CORESMITH_OUTPUT_CONTRACT_GATE", "0")
        # both gates off -> continue goes to Interface Definition (legacy)
        assert ag.route_after_diagram_escalation(
            {"human_response": {"action": "continue"}}) == "Interface Definition"


class TestAlgoAxisLocFloor:
    def test_small_multi_algo_store_not_flagged(self):
        # A small block touching many algorithms (a hoisted store) must NOT
        # flag on the algo axis alone -- only a SUBSTANTIAL block does.
        assert bc.MODELING_ALGO_LOC_FLOOR > 0
        # single tiny function that trips distinct-algorithm detection but is
        # far below the LOC floor -> no breach.
        stats = _stats()
        est = bc.estimate_block_complexity(
            "coefficient_token_memory", "g.py", stats=stats,
            slice_fns=["store_put"])
        assert est["over_budget"] is False
