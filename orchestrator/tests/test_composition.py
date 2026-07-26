# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the RETIRED v1 deterministic block-golden composition logic.

No LLM, no EDA. Compatible with `-m "not live_llm and not requires_nix and not e2e"`.

The v1 ``compose_and_run`` harness + ``BlockGolden.step()`` contract + FRD FUNC
vectors are superseded by the v2 MyHDL block-model + model-integration gate
(see test_model_integration.py). The v1 helpers are kept in composition.py for
forensic value, and these tests pin their behaviour; the end-to-end v1 gate is
now reached via the retained ``_run_composition_gate_v1`` (the public
``run_composition_gate`` delegates to the v2 model-integration gate).

Still-relevant shared helpers exercised here: resolve_reference_implementation,
resolve_reference_entrypoint, _run_reference signature mapping, and the
``block_goldens_enabled`` env flag.
"""

from __future__ import annotations

import textwrap
import types as _types
from pathlib import Path

import pytest

from orchestrator.architecture import composition

# ---------------------------------------------------------------------------
# Toy block goldens (written to disk, loaded via importlib like real ones)
# ---------------------------------------------------------------------------

BLOCK_A_DOUBLER = '''\
PORTS = {"inputs": ["in"], "outputs": ["out"]}


class BlockGolden:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, inputs):
        if "in" not in inputs:
            return {}
        return {"out": inputs["in"] * 2}
'''

BLOCK_B_ADD1 = '''\
PORTS = {"inputs": ["in"], "outputs": ["out"]}


class BlockGolden:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, inputs):
        if "in" not in inputs:
            return {}
        return {"out": inputs["in"] + 1}
'''

# Deliberately wrong: adds 99 instead of 1.
BLOCK_B_WRONG = '''\
PORTS = {"inputs": ["in"], "outputs": ["out"]}


class BlockGolden:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, inputs):
        if "in" not in inputs:
            return {}
        return {"out": inputs["in"] + 99}
'''

# Accumulator with a feedback edge from its own previous output.
BLOCK_ACC = '''\
PORTS = {"inputs": ["in", "acc_fb"], "outputs": ["out"]}


class BlockGolden:
    def __init__(self):
        self.last = 0

    def reset(self):
        self.last = 0

    def step(self, inputs):
        prev = inputs.get("acc_fb", 0)
        val = inputs.get("in", 0)
        total = prev + val
        return {"out": total}
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Toy chip = blockB(blockA(x)) = (x*2) + 1
TOY_BLOCK_DIAGRAM = {
    "blocks": [
        {"name": "blockA", "interfaces": {"in": {"direction": "input"},
                                          "out": {"direction": "output"}}},
        {"name": "blockB", "interfaces": {"in": {"direction": "input"},
                                          "out": {"direction": "output"}}},
    ],
    "connections": [
        # chip ingress -> blockA.in
        {"from": "chip_in", "to": "blockA", "from_port": "chip_in", "to_port": "in"},
        # blockA.out -> blockB.in
        {"from": "blockA", "to": "blockB", "from_port": "out", "to_port": "in"},
        # blockB.out -> chip egress
        {"from": "blockB", "to": "chip_out", "from_port": "out", "to_port": "chip_out"},
    ],
}


def toy_reference(x: int) -> int:
    """The reference implementation: (x*2) + 1."""
    return x * 2 + 1


# ---------------------------------------------------------------------------
# compose_and_run -- forward DAG
# ---------------------------------------------------------------------------

class TestComposeAndRun:
    def test_matches_reference_for_several_inputs(self, tmp_path):
        d = tmp_path / "arch" / "block_goldens"
        _write(d / "blockA.py", BLOCK_A_DOUBLER)
        _write(d / "blockB.py", BLOCK_B_ADD1)
        goldens = composition.load_block_goldens(str(d), ["blockA", "blockB"])

        for x in (0, 1, 3, 7, 42, 255):
            out = composition.compose_and_run(
                TOY_BLOCK_DIAGRAM, goldens, {"chip_in": [x]}
            )
            assert out["chip_out"] == [toy_reference(x)], (x, out)

    def test_multi_transaction_stream(self, tmp_path):
        d = tmp_path / "arch" / "block_goldens"
        _write(d / "blockA.py", BLOCK_A_DOUBLER)
        _write(d / "blockB.py", BLOCK_B_ADD1)
        goldens = composition.load_block_goldens(str(d), ["blockA", "blockB"])

        xs = [1, 2, 3, 4]
        out = composition.compose_and_run(
            TOY_BLOCK_DIAGRAM, goldens, {"chip_in": xs}
        )
        assert out["chip_out"] == [toy_reference(x) for x in xs]

    def test_wrong_block_diverges(self, tmp_path):
        d = tmp_path / "arch" / "block_goldens"
        _write(d / "blockA.py", BLOCK_A_DOUBLER)
        _write(d / "blockB.py", BLOCK_B_WRONG)
        goldens = composition.load_block_goldens(str(d), ["blockA", "blockB"])

        out = composition.compose_and_run(
            TOY_BLOCK_DIAGRAM, goldens, {"chip_in": [5]}
        )
        # wrong: (5*2)+99 = 109 ; reference (5*2)+1 = 11
        assert out["chip_out"] == [109]
        assert out["chip_out"] != [toy_reference(5)]


# ---------------------------------------------------------------------------
# Feedback edge (one-transaction delay accumulator)
# ---------------------------------------------------------------------------

class TestFeedbackEdge:
    def test_accumulator_one_txn_delay(self, tmp_path):
        d = tmp_path / "arch" / "block_goldens"
        _write(d / "acc.py", BLOCK_ACC)
        goldens = composition.load_block_goldens(str(d), ["acc"])

        bd = {
            "blocks": [
                {"name": "acc", "interfaces": {
                    "in": {"direction": "input"},
                    "acc_fb": {"direction": "input"},
                    "out": {"direction": "output"},
                }},
            ],
            "connections": [
                {"from": "chip_in", "to": "acc",
                 "from_port": "chip_in", "to_port": "in"},
                # feedback: acc.out -> acc.acc_fb (self-loop, one-txn delay)
                {"from": "acc", "to": "acc",
                 "from_port": "out", "to_port": "acc_fb"},
                {"from": "acc", "to": "chip_out",
                 "from_port": "out", "to_port": "chip_out"},
            ],
        }

        # Feed [1,2,3,4]: with one-txn-delay feedback the running total is the
        # prefix sum -> 1, 3, 6, 10.
        out = composition.compose_and_run(bd, goldens, {"chip_in": [1, 2, 3, 4]})
        assert out["chip_out"] == [1, 3, 6, 10]


# ---------------------------------------------------------------------------
# parse_func_vectors
# ---------------------------------------------------------------------------

class TestParseFuncVectors:
    def test_small_snippet(self):
        frd = textwrap.dedent(
            """\
            # FRD

            ## Functional Vectors

            - **ID**: FUNC-001
            - **Block / I-O ports**: blockA / in -> out
            - **Stimulus**: 5
            - **Expected output**: 11
            - **Priority**: must_have

            - **ID**: FUNC-002
            - **Block / I-O ports**: blockB / in -> out
            - **Stimulus**: 3
            - **Expected output**: 7
            - **Priority**: should_have

            ## Timing Requirements
            - **ID**: TIME-001
            """
        )
        vecs = composition.parse_func_vectors(frd)
        ids = {v["id"] for v in vecs}
        assert ids == {"FUNC-001", "FUNC-002"}
        by_id = {v["id"]: v for v in vecs}
        assert by_id["FUNC-001"]["block"].startswith("blockA")
        assert by_id["FUNC-001"]["stimulus"] == 5
        assert by_id["FUNC-001"]["expected_output"] == 11
        assert by_id["FUNC-001"]["priority"] == "must_have"
        # TIME-001 must NOT be parsed as a FUNC vector.
        assert "TIME-001" not in str(ids)

    def test_real_frd_field_layout(self):
        # Mirrors the field labels from orchestrator/langchain/prompts/frd_spec.md
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            ### FUNC-010
            - **ID**: FUNC-010
            - **Block / I-O ports**: dct_engine / pixel_in -> coeff_out
            - **Stimulus**: [8, 8, 8, 8, 8, 8, 8, 8]
            - **Expected output**: [64, 0, 0, 0, 0, 0, 0, 0]
            - **Priority**: must_have
            """
        )
        vecs = composition.parse_func_vectors(frd)
        assert len(vecs) == 1
        v = vecs[0]
        assert v["id"] == "FUNC-010"
        assert v["stimulus"] == [8, 8, 8, 8, 8, 8, 8, 8]
        assert v["expected_output"] == [64, 0, 0, 0, 0, 0, 0, 0]

    def test_empty_returns_empty(self):
        assert composition.parse_func_vectors("") == []
        assert composition.parse_func_vectors("# no vectors here") == []


# ---------------------------------------------------------------------------
# run_composition_gate -- end to end on a toy project tree
# ---------------------------------------------------------------------------

def _toy_project(tmp_path: Path, block_b_text: str, *, with_reference=True):
    """Build a minimal project tree the gate can consume."""
    import json

    root = tmp_path
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / ".coresmith" / "block_diagram.json").write_text(
        json.dumps(TOY_BLOCK_DIAGRAM), encoding="utf-8"
    )

    gd = root / "arch" / "block_goldens"
    _write(gd / "blockA.py", BLOCK_A_DOUBLER)
    _write(gd / "blockB.py", block_b_text)

    frd = textwrap.dedent(
        """\
        ## Functional Vectors

        - **ID**: FUNC-001
        - **Block / I-O ports**: blockB / chip_in -> chip_out
        - **Stimulus**: {"chip_in": [5]}
        - **Expected output**: 11
        - **Priority**: must_have
        """
    )
    _write(root / "arch" / "frd_spec.md", frd)

    if with_reference:
        ref = textwrap.dedent(
            """\
            def run(stim):
                x = stim["chip_in"][0]
                return x * 2 + 1
            """
        )
        _write(root / "inputs" / "toy_golden.py", ref)
    return root


class TestRunCompositionGate:
    def test_noop_when_flag_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        root = _toy_project(tmp_path, BLOCK_B_WRONG)  # wrong, but flag off
        assert composition.run_composition_gate(str(root)) == []

    def test_passes_when_correct(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        root = _toy_project(tmp_path, BLOCK_B_ADD1)
        violations = composition.run_composition_gate(str(root))
        assert violations == [], violations

    def test_flags_wrong_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        root = _toy_project(tmp_path, BLOCK_B_WRONG)
        # v2 run_composition_gate delegates to the MyHDL model-integration gate;
        # the retired v1 composition logic is exercised via _run_composition_gate_v1.
        violations = composition._run_composition_gate_v1(str(root))
        assert violations, "expected a composition violation"
        v = violations[0]
        assert v["type"] == "composition_gate_failure"
        # The FUNC vector names blockB as its block, so divergence localizes there.
        assert v["first_divergence_block"] == "blockB"
        assert v["vector_id"] == "FUNC-001"

    def test_noop_when_no_reference(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        # No reference file -> still uses FUNC expected output for comparison,
        # but resolve_reference_implementation must return None here. Build a
        # tree with NO *_golden.py anywhere; the gate then relies on FUNC
        # expected output. To exercise the "no reference => no-op" branch we
        # remove the FUNC section too.
        import json
        root = tmp_path
        (root / ".coresmith").mkdir(parents=True, exist_ok=True)
        (root / ".coresmith" / "block_diagram.json").write_text(
            json.dumps(TOY_BLOCK_DIAGRAM), encoding="utf-8"
        )
        gd = root / "arch" / "block_goldens"
        _write(gd / "blockA.py", BLOCK_A_DOUBLER)
        _write(gd / "blockB.py", BLOCK_B_WRONG)
        # No frd, no reference impl.
        assert composition.run_composition_gate(str(root)) == []

    def test_noop_when_no_goldens_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        import json
        root = tmp_path
        (root / ".coresmith").mkdir(parents=True, exist_ok=True)
        (root / ".coresmith" / "block_diagram.json").write_text(
            json.dumps(TOY_BLOCK_DIAGRAM), encoding="utf-8"
        )
        assert composition.run_composition_gate(str(root)) == []


# ---------------------------------------------------------------------------
# resolve_reference_implementation
# ---------------------------------------------------------------------------

class TestResolveReference:
    def test_finds_inputs_golden(self, tmp_path):
        ref = tmp_path / "inputs" / "mydesign_golden.py"
        _write(ref, "def run(x): return x\n")
        found = composition.resolve_reference_implementation(str(tmp_path))
        assert found is not None
        assert Path(found).name == "mydesign_golden.py"

    def test_none_when_absent(self, tmp_path):
        assert composition.resolve_reference_implementation(str(tmp_path)) is None

    def test_env_source_root_wins_over_prd_golden(self, tmp_path, monkeypatch):
        """An explicit CORESMITH_SOURCE_ROOT override must beat auto-discovery
        (e.g. a *_golden.py named in the PRD) so a bitstream-only reference
        WRAPPER can be selected even when the PRD cites the raw golden."""
        # PRD names a golden that also exists on disk (would win at priority 1).
        prd_golden = tmp_path / "examples" / "d" / "thing_golden.py"
        _write(prd_golden, "def encode_image(x): return (b'', [], None)\n")
        _write(tmp_path / "arch" / "prd_spec.md",
               "reference: examples/d/thing_golden.py\n")
        # Operator points at a bitstream-only wrapper instead.
        wrapper = tmp_path / "ref_wrapper.py"
        _write(wrapper, "def encode(x, qp=36): return b''\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(wrapper))
        found = composition.resolve_reference_implementation(str(tmp_path))
        assert Path(found).name == "ref_wrapper.py", found

    def test_prd_golden_used_when_no_env_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        golden = tmp_path / "examples" / "d" / "thing_golden.py"
        _write(golden, "def run(x): return x\n")
        found = composition.resolve_reference_implementation(str(tmp_path))
        assert Path(found).name == "thing_golden.py", found

    def test_generator_reference_distinct_from_gate(self, tmp_path, monkeypatch):
        """Generators get the FULL golden via CORESMITH_GENERATOR_SOURCE even when
        the gate's CORESMITH_SOURCE_ROOT points at a bytes-only wrapper."""
        full = tmp_path / "design_golden.py"
        _write(full, "def encode_image_v2(p, qp=36): return (b'', [], None)\n")
        wrapper = tmp_path / "wrap.py"
        _write(wrapper, "def encode(p, qp=36): return b''\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT", str(wrapper))      # gate oracle
        monkeypatch.setenv("CORESMITH_GENERATOR_SOURCE", str(full))     # generators
        assert Path(composition.resolve_reference_implementation(str(tmp_path))).name == "wrap.py"
        assert Path(composition.resolve_generator_reference(str(tmp_path))).name == "design_golden.py"

    def test_generator_reference_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_GENERATOR_SOURCE", raising=False)
        monkeypatch.delenv("CORESMITH_SOURCE_ROOT", raising=False)
        golden = tmp_path / "inputs" / "x_golden.py"
        _write(golden, "def run(x): return x\n")
        # No generator override -> same as resolve_reference_implementation.
        assert (composition.resolve_generator_reference(str(tmp_path))
                == composition.resolve_reference_implementation(str(tmp_path)))


# ---------------------------------------------------------------------------
# block_goldens_enabled flag helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# resolve_reference_entrypoint
# ---------------------------------------------------------------------------


def _make_ref_module(src: str, name: str = "_test_ref_mod"):
    """Build a module object from source for entrypoint tests."""
    mod = _types.ModuleType(name)
    mod.__name__ = name
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


class TestResolveReferenceEntrypoint:
    def test_env_bare_func(self, tmp_path, monkeypatch):
        mod = _make_ref_module("def my_entry(x):\n    return x + 1\n")
        monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "my_entry")
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert callable(fn)
        assert name == "my_entry"
        assert fn(4) == 5

    def test_env_module_colon_func(self, tmp_path, monkeypatch):
        # module:func form resolving via attr fallback on the ref module.
        mod = _make_ref_module("def top(x):\n    return x * 3\n")
        # "doesnotexist:top" -> import fails, falls back to getattr(mod, top)
        monkeypatch.setenv("CORESMITH_REFERENCE_ENTRY", "nosuchmod:top")
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert callable(fn)
        assert fn(2) == 6

    def test_declared_in_prd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        mod = _make_ref_module(
            "def helper(x):\n    return x\n"
            "def the_oracle(x):\n    return x - 7\n"
        )
        (tmp_path / "arch").mkdir(parents=True, exist_ok=True)
        (tmp_path / "arch" / "prd_spec.md").write_text(
            "Notes\nreference_entry_point: the_oracle\nmore\n",
            encoding="utf-8",
        )
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert name == "the_oracle"
        assert fn(10) == 3

    def test_discovery_preferred_name(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        mod = _make_ref_module(
            "def aux(x):\n    return 0\n"
            "def encode(x):\n    return x + 100\n"
        )
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert name == "encode"
        assert fn(1) == 101

    def test_discovery_single_function(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        mod = _make_ref_module("def whatever(x):\n    return x * x\n")
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert name == "whatever"
        assert fn(5) == 25

    def test_discovery_ambiguous_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        # Two non-conventional public functions, no clear winner -> None.
        mod = _make_ref_module("def alpha(x):\n    return 1\ndef beta(x):\n    return 2\n")
        fn, name = composition.resolve_reference_entrypoint(str(tmp_path), mod)
        assert fn is None


# ---------------------------------------------------------------------------
# _run_reference signature mapping
# ---------------------------------------------------------------------------

class TestRunReferenceSignatureMapping:
    def test_by_name_mapping(self):
        def ref(a, b):
            return a * 10 + b

        out = composition._run_reference(ref, {"a": 3, "b": 4})
        assert out == 34

    def test_positional_when_names_dont_match(self):
        def ref(x, y):
            return [x, y]

        # keys do not match params -> positional in insertion order.
        out = composition._run_reference(ref, {"port_one": 1, "port_two": 2})
        assert out == [1, 2]

    def test_single_param_takes_whole_dict(self):
        def ref(stim):
            return stim["chip_in"][0] * 2 + 1

        out = composition._run_reference(ref, {"chip_in": [5]})
        assert out == 11

    def test_failure_returns_none(self):
        def ref(a, b, c):
            return a + b + c

        # too few args available -> exception -> None
        out = composition._run_reference(ref, {"only_one": 1})
        assert out is None

    def test_failure_reraises_when_requested(self):
        def ref(x):
            return x.shape  # crashes on a list -> no oracle

        # Default: swallow -> None (legacy/v1 path).
        assert composition._run_reference(ref, [1, 2, 3]) is None
        # reraise=True (the gate): a crash must surface, not silently pass.
        with pytest.raises(composition.ReferenceInvocationError):
            composition._run_reference(ref, [1, 2, 3], reraise=True)

    def test_tuple_normalized_to_list(self):
        def ref(x):
            return (x, x + 1)

        assert composition._run_reference(ref, 5) == [5, 6]


# ---------------------------------------------------------------------------
# parse_func_vectors -- structured stimulus/expected
# ---------------------------------------------------------------------------

class TestParseFuncVectorsStructured:
    def test_fenced_json_stimulus_and_expected(self):
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            ### FUNC-001
            - **ID**: FUNC-001
            - **Block / I-O ports**: top / chip_in -> chip_out
            - **Stimulus**: some prose describing the input
            - **Expected output**: prose value
            - **Machine-readable vector**:

            ```json
            {"stimulus": {"chip_in": [1, 2, 3]}, "expected": {"chip_out": [3, 5, 7]}}
            ```
            - **Priority**: must_have
            """
        )
        vecs = composition.parse_func_vectors(frd)
        assert len(vecs) == 1
        v = vecs[0]
        assert v["stimulus_struct"] == {"chip_in": [1, 2, 3]}
        assert v["expected_struct"] == {"chip_out": [3, 5, 7]}
        # prose fields preserved for back-compat
        assert v["stimulus"]

    def test_fenced_json_stimulus_only(self):
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            ### FUNC-002
            - **ID**: FUNC-002
            ```json
            {"stimulus": {"in": [9]}}
            ```
            """
        )
        vecs = composition.parse_func_vectors(frd)
        v = vecs[0]
        assert v["stimulus_struct"] == {"in": [9]}
        assert v["expected_struct"] is None

    def test_inline_structured_stimulus(self):
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            ### FUNC-003
            - **ID**: FUNC-003
            - **Stimulus (structured)**: {"pixels": [8, 8]}
            """
        )
        vecs = composition.parse_func_vectors(frd)
        v = vecs[0]
        assert v["stimulus_struct"] == {"pixels": [8, 8]}

    def test_no_structured_block_leaves_none(self):
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            - **ID**: FUNC-004
            - **Stimulus**: 5
            - **Expected output**: 11
            """
        )
        vecs = composition.parse_func_vectors(frd)
        v = vecs[0]
        assert v["stimulus_struct"] is None
        assert v["expected_struct"] is None
        assert v["stimulus"] == 5  # prose path unchanged


# ---------------------------------------------------------------------------
# Reference-as-oracle differential gate (end to end, no LLM/EDA)
# ---------------------------------------------------------------------------

def _ref_oracle_project(tmp_path: Path, block_b_text: str):
    """Toy project where the REFERENCE (not prose) is the oracle.

    chip = blockB(blockA(x)); reference ref(x) = x*2+1. FRD vector carries a
    structured stimulus and NO expected (reference computes it).
    """
    import json as _json

    root = tmp_path
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / ".coresmith" / "block_diagram.json").write_text(
        _json.dumps(TOY_BLOCK_DIAGRAM), encoding="utf-8"
    )
    gd = root / "arch" / "block_goldens"
    _write(gd / "blockA.py", BLOCK_A_DOUBLER)
    _write(gd / "blockB.py", block_b_text)

    frd = textwrap.dedent(
        """\
        ## Functional Vectors

        ### FUNC-001
        - **ID**: FUNC-001
        - **Block / I-O ports**: blockB / chip_in -> chip_out
        - **Stimulus**: prose
        - **Machine-readable vector**:
        ```json
        {"stimulus": {"chip_in": [1, 2, 3]}}
        ```
        - **Priority**: must_have
        """
    )
    _write(root / "arch" / "frd_spec.md", frd)

    # Reference implementation: x*2+1, single-param entry "run".
    ref = textwrap.dedent(
        """\
        def run(stim):
            xs = stim["chip_in"]
            return [x * 2 + 1 for x in xs]
        """
    )
    _write(root / "inputs" / "toy_golden.py", ref)
    return root


class TestReferenceAsOracleGate:
    def test_passes_when_composition_matches_reference(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        root = _ref_oracle_project(tmp_path, BLOCK_B_ADD1)
        violations = composition.run_composition_gate(str(root))
        assert violations == [], violations

    def test_flags_planted_wrong_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        # blockB adds 99 -> composed = x*2+99, reference = x*2+1 -> divergence.
        root = _ref_oracle_project(tmp_path, BLOCK_B_WRONG)
        violations = composition._run_composition_gate_v1(str(root))
        assert violations, "expected a reference-vs-composition violation"
        v = violations[0]
        assert v["type"] == "composition_gate_failure"
        assert v["first_divergence_block"] == "blockB"
        assert v["vector_id"] == "FUNC-001"
        # Expected comes from the reference (x*2+1 for [1,2,3]).
        assert v["expected"] == [3, 5, 7]
        assert "reference" in v["suggested_fix"].lower()


class TestNoReferenceFallback:
    def test_uses_expected_struct_when_no_entry(self, tmp_path, monkeypatch):
        import json as _json

        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_REFERENCE_ENTRY", raising=False)
        # No reference impl anywhere; rely on structured expected. blockB wrong.
        root = tmp_path
        (root / ".coresmith").mkdir(parents=True, exist_ok=True)
        (root / ".coresmith" / "block_diagram.json").write_text(
            _json.dumps(TOY_BLOCK_DIAGRAM), encoding="utf-8"
        )
        gd = root / "arch" / "block_goldens"
        _write(gd / "blockA.py", BLOCK_A_DOUBLER)
        _write(gd / "blockB.py", BLOCK_B_WRONG)
        frd = textwrap.dedent(
            """\
            ## Functional Vectors

            ### FUNC-001
            - **ID**: FUNC-001
            - **Block / I-O ports**: blockB / chip_in -> chip_out
            ```json
            {"stimulus": {"chip_in": [5]}, "expected": {"chip_out": [11]}}
            ```
            """
        )
        _write(root / "arch" / "frd_spec.md", frd)
        # resolve_reference_implementation must find nothing.
        assert composition.resolve_reference_implementation(str(root)) is None
        violations = composition._run_composition_gate_v1(str(root))
        # composed = 5*2+99 = 109 != expected_struct 11 -> violation.
        assert violations, "expected fallback expected_struct to flag mismatch"
        assert violations[0]["expected"] in ({"chip_out": [11]}, [11], 11)


class TestFlagHelper:
    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "On"])
    def test_truthy(self, monkeypatch, val):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", val)
        assert composition.block_goldens_enabled() is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off"])
    def test_falsy(self, monkeypatch, val):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", val)
        assert composition.block_goldens_enabled() is False

    def test_unset(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        assert composition.block_goldens_enabled() is False


class TestFloatPolicyHelpers:
    """Fixed-point-default / float-epsilon gate policy (user spec): bias to
    fixed-point bit-exact; when the reference OUTPUT is float-valued, allow an
    epsilon tolerance."""

    def test_output_has_float_detection(self):
        assert composition.output_has_float([1, 2, 3]) is False        # ints
        assert composition.output_has_float([1, 2.0, 3]) is True       # a float
        assert composition.output_has_float({"a": [0, 0]}) is False
        assert composition.output_has_float({"a": [0, 0.5]}) is True
        assert composition.output_has_float(b"\x01\x02") is False      # bytes
        assert composition.output_has_float([True, False]) is False    # bools
        import numpy as np
        assert composition.output_has_float(np.array([1, 2], dtype=np.int32)) is False
        assert composition.output_has_float(np.array([1.0], dtype=np.float64)) is True

    def test_gate_epsilon_default_and_override(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_GATE_EPSILON", raising=False)
        assert composition.gate_epsilon() == 1e-6
        monkeypatch.setenv("CORESMITH_GATE_EPSILON", "1e-3")
        assert composition.gate_epsilon() == 1e-3
        monkeypatch.setenv("CORESMITH_GATE_EPSILON", "garbage")
        assert composition.gate_epsilon() == 1e-6

    def test_outputs_close_within_and_outside_eps(self):
        assert composition.outputs_close([1.0, 2.0], [1.0, 2.0000001], 1e-5) is True
        assert composition.outputs_close([1.0, 2.0], [1.0, 2.5], 1e-5) is False
        # length mismatch -> not close
        assert composition.outputs_close([1.0], [1.0, 2.0], 1e-5) is False
        # non-numeric leaves must be exactly equal
        assert composition.outputs_close(["a", 1.0], ["a", 1.0], 1e-5) is True
        assert composition.outputs_close(["a", 1.0], ["b", 1.0], 1e-5) is False

    def test_int_output_unaffected_by_eps(self):
        # Integer (byte) outputs are compared exactly elsewhere; outputs_close
        # with eps=0 still requires equality for ints.
        assert composition.outputs_close([129, 39], [129, 39], 0.0) is True
        assert composition.outputs_close([129, 39], [129, 40], 0.0) is False
