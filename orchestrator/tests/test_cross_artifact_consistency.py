# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Architecture-phase cross-artifact consistency gate.

The acceptance set is five real contradictions from one architecture run, each
of which was caught one-at-a-time by a *downstream* gate days later, each at
the cost of a full re-spec:

  1. FRD vs ERS -- FRD said 12.5 MHz / 50 Mbit/s / 40 ns phases, ERS said
     6.25 MHz / 25 Mbit/s / 80 ns.                     (deterministic half)
  2. ERS vs itself -- the READ-serializer requirement mandates rise-scheduled
     nibble presentation, the per-block requirement says the serializer
     advances on sck_fall.                             (LLM half)
  3. ERS vs interface_contracts -- qspi_drive.rate_description carries the
     fall-scheduling wording.                          (LLM half)
  4. ERS vs interface_contracts -- edge_event nibble_sampling likewise.
                                                       (LLM half)
  5. ERS vs block_diagram -- the qspi_drive connection's semantic_contract
     likewise.                                         (LLM half)

The fixtures under ``fixtures/cross_artifact/`` carry the contradictions
verbatim (copied from the run's ``.bak-chiplead*`` before-files) plus a
``clean/`` counterpart with every correction applied, so a false positive on
consistent artifacts fails a test.

Hermetic: no LLM is called except through a monkeypatched ``ClaudeLLM``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.architecture import constraints as cmod
from orchestrator.architecture import cross_artifact as ca

FIXTURES = Path(__file__).parent / "fixtures" / "cross_artifact"


# ---------------------------------------------------------------------------
# Fixture plumbing
# ---------------------------------------------------------------------------

def _materialize(flavor: str, dest: Path) -> Path:
    """Copy one fixture flavor into a throwaway project root."""
    src = FIXTURES / flavor
    assert src.is_dir(), f"missing fixture flavor: {src}"
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


@pytest.fixture
def contradictory_project(tmp_path) -> Path:
    return _materialize("contradictory", tmp_path / "contradictory")


@pytest.fixture
def clean_project(tmp_path) -> Path:
    return _materialize("clean", tmp_path / "clean")


def _load(root: Path) -> tuple[dict, dict]:
    bd = json.loads((root / ".coresmith" / "block_diagram.json").read_text())
    ic = json.loads(
        (root / ".coresmith" / "interface_contracts.json").read_text()
    )
    return bd, ic


def _deterministic(root: Path) -> list[dict]:
    bd, ic = _load(root)
    return cmod._check_cross_artifact_quantities(
        project_root=str(root), block_diagram=bd, interface_contracts=ic,
    )


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# The fixtures really do carry the five contradictions
# ---------------------------------------------------------------------------

class TestFixturesCarryTheAcceptanceSet:
    """If these drift, every test below is testing nothing."""

    def test_1_frd_ers_rate_disagreement(self, contradictory_project):
        frd = (contradictory_project / "arch" / "frd_spec.md").read_text()
        ers = (contradictory_project / "arch" / "ers_spec.md").read_text()
        assert "12.5 MHz" in frd and "50 Mbit/s" in frd and "40 ns" in frd
        assert "6.25 MHz" in ers and "25 Mbit/s" in ers and "80 ns" in ers

    def test_2_ers_contradicts_itself(self, contradictory_project):
        ers = (contradictory_project / "arch" / "ers_spec.md").read_text()
        assert "PRECEDING SYNCHRONIZED RISING-SCK EVENT" in ers
        assert "every sck_fall in data phase updates one nibble" in ers

    def test_3_and_4_contracts_carry_fall_scheduling(
        self, contradictory_project,
    ):
        _bd, ic = _load(contradictory_project)
        by_port = {c["producer_port"]: c for c in ic["contracts"]}
        assert (
            "only on synchronized SCK falling-edge events"
            in by_port["qspi_drive"]["rate_description"]
        )
        rules = by_port["edge_event"]["representations"]["state_semantics"]
        nibble = next(r for r in rules if r["name"] == "nibble_sampling")
        assert "on sck_fall=1 it advances only the read serializer" in nibble["rule"]

    def test_5_block_diagram_carries_fall_scheduling(
        self, contradictory_project,
    ):
        bd, _ic = _load(contradictory_project)
        conn = next(
            c for c in bd["connections"] if c["interface"] == "qspi_drive"
        )
        assert "synchronized falling-SCK update" in conn["semantic_contract"]

    def test_clean_flavor_has_all_five_corrected(self, clean_project):
        frd = (clean_project / "arch" / "frd_spec.md").read_text()
        ers = (clean_project / "arch" / "ers_spec.md").read_text()
        bd, ic = _load(clean_project)
        assert "12.5 MHz" not in frd and "50 Mbit/s" not in frd
        assert "40 ns" not in frd
        assert "every sck_fall in data phase updates one nibble" not in ers
        assert "synchronized falling-SCK update" not in json.dumps(bd)
        assert "only on synchronized SCK falling-edge events" not in json.dumps(ic)


# ---------------------------------------------------------------------------
# Deterministic half -- contradiction 1
# ---------------------------------------------------------------------------

class TestDeterministicQuantityCheck:

    def test_catches_contradiction_1(self, contradictory_project):
        """The FRD/ERS operating-point drift is caught arithmetically."""
        violations = _deterministic(contradictory_project)
        assert violations, "contradiction 1 was not caught"
        for v in violations:
            assert v["check"] == "cross_artifact_consistency"
            assert v["category"] == "structural"
            assert v["severity"] == "error"
            assert v["finding_kind"] == "deterministic_quantity"

        # Both halves of the drift: the SCK rate cap and the phase-width floor.
        named = {(v["quantity_name"], v["dimension"]) for v in violations}
        assert ("sck", "frequency") in named
        assert ("tlow", "time") in named

        freq = next(v for v in violations if v["dimension"] == "frequency")
        assert "12.5 MHz" in freq["violation"]
        assert "6.25 MHz" in freq["violation"]

        time_v = next(v for v in violations if v["dimension"] == "time")
        assert "40 ns" in time_v["violation"]
        assert "80 ns" in time_v["violation"]

    def test_finding_names_both_sides_with_file_and_location(
        self, contradictory_project,
    ):
        """A human must be able to open both sides without re-deriving them."""
        for v in _deterministic(contradictory_project):
            locs = v["locations"]
            assert len(locs) == 2
            assert {loc["file"] for loc in locs} == {
                "arch/frd_spec.md", "arch/ers_spec.md",
            }
            for loc in locs:
                assert loc["location"].startswith("line ")
                assert loc["quote"]
            assert v["evidence"].startswith("A: ")
            assert "\nB: " in v["evidence"]

    def test_no_false_positive_on_corrected_artifacts(self, clean_project):
        """The negative test: consistent artifacts must produce nothing."""
        assert _deterministic(clean_project) == []

    def test_env_gate_off_disables(self, contradictory_project, monkeypatch):
        monkeypatch.setenv("CORESMITH_CROSS_ARTIFACT_GATE", "0")
        assert _deterministic(contradictory_project) == []

    def test_env_gate_defaults_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_CROSS_ARTIFACT_GATE", raising=False)
        assert cmod._cross_artifact_gate_enabled() is True

    def test_needs_two_artifacts(self, tmp_path):
        """One artifact cannot contradict anything -- no-op, not a crash."""
        (tmp_path / ".coresmith").mkdir()
        bd = json.loads(
            (FIXTURES / "contradictory" / ".coresmith"
             / "block_diagram.json").read_text()
        )
        assert cmod._check_cross_artifact_quantities(
            project_root=str(tmp_path), block_diagram=bd,
            interface_contracts={},
        ) == []


# ---------------------------------------------------------------------------
# Deterministic half -- the "never guess" contract
# ---------------------------------------------------------------------------

class TestDeterministicNeverGuesses:
    """A quantity that cannot be named confidently is SKIPPED WITH A NOTE."""

    VOCAB = frozenset({"sck", "qspi_sck", "qspi", "fifo"})

    def _extract(self, text):
        notes: list[str] = []
        got = ca.extract_quantities(text, "ers", "line 1", self.VOCAB, notes)
        return got, notes

    @pytest.mark.parametrize("text,why", [
        ("The link runs at 12.5 GB/s of payload.", "ambiguous unit"),
        ("CDC of a falling edge costs ~30 ns of the low phase.", "approximate"),
        ("SCK operates from 1 to 6.25 MHz across the range.", "range"),
        ("Legal offsets are 6.25, 5, and 1 MHz SCK.", "list member"),
        ("The measured value of 12.5 MHz was observed.", "unnamed"),
    ])
    def test_unconfident_quantities_are_noted_not_flagged(self, text, why):
        got, notes = self._extract(text)
        assert got == [], f"{why!r} case should not have been extracted: {got}"
        assert notes, f"{why!r} case should have produced a skip note"
        assert "skipped" in notes[0]

    @pytest.mark.parametrize("text,name,value,cmp_", [
        ("SCK `<=6.25 MHz` is the cap.", "sck", 6.25, "le"),
        ("Constrain tHIGH>=80 ns at the pin.", "thigh", 80.0, "ge"),
        ("At qspi_sck=6.25 MHz the link is legal.", "qspi_sck", 6.25, "eq"),
        ("Runs at 12.5 MHz QSPI in the worst case.", "qspi", 12.5, "eq"),
    ])
    def test_adjacently_named_quantities_are_extracted(
        self, text, name, value, cmp_,
    ):
        got, _notes = self._extract(text)
        assert len(got) == 1, got
        assert got[0].name == name
        assert got[0].raw_value == value
        assert got[0].comparator == cmp_

    def test_anti_pattern_sentences_are_not_claims(self):
        got, _ = self._extract(
            "Do not run SCK `<=12.5 MHz`; that is the forbidden pattern."
        )
        assert got == []

    def test_units_normalize_before_comparison(self):
        """6.25 MHz and 6250 kHz are the same claim, not a contradiction."""
        notes: list[str] = []
        a = ca.extract_quantities(
            "SCK `<=6.25 MHz` cap.", "ers", "line 1", self.VOCAB, notes)
        b = ca.extract_quantities(
            "SCK `<=6250 kHz` cap.", "frd", "line 1", self.VOCAB, notes)
        assert ca.find_quantity_conflicts(a + b) == []

    def test_intra_artifact_disagreement_is_not_this_half(self):
        """Two values in ONE artifact are the LLM half's problem."""
        notes: list[str] = []
        qs = ca.extract_quantities(
            "SCK `<=6.25 MHz` cap.", "ers", "line 1", self.VOCAB, notes,
        ) + ca.extract_quantities(
            "SCK `<=12.5 MHz` cap.", "ers", "line 9", self.VOCAB, notes,
        )
        assert ca.find_quantity_conflicts(qs) == []

    def test_prose_cannot_invent_an_anchor(self):
        """Names come from the structured artifacts or an identifier shape."""
        vocab = ca.build_anchor_vocabulary(
            block_diagram={"blocks": [{"name": "qspi_slave_engine",
                                       "interfaces": {"qspi_drive": {}}}]},
        )
        assert "qspi_slave_engine" in vocab and "qspi" in vocab
        # Generic words never become names.
        assert "block" not in vocab and "drive" in vocab
        assert ca._is_named_quantity("throughput", vocab) is False
        assert ca._is_named_quantity("The", vocab) is False
        assert ca._is_named_quantity("SCK", vocab) is True


# ---------------------------------------------------------------------------
# LLM half -- contradictions 2..5
# ---------------------------------------------------------------------------

def _subagent_stub(per_check_response: dict) -> AsyncMock:
    """Same shape as the existing constraint-subagent stubs in test_architecture."""
    async def _fake_call(*_args, **kwargs):
        run_name = kwargs.get("run_name", "")
        check_id = run_name.split(":", 1)[1] if ":" in run_name else ""
        return json.dumps(
            per_check_response.get(check_id, {"pass": True, "evidence": "n/a"})
        )

    return AsyncMock(side_effect=_fake_call)


# The four semantic contradictions, in the shape the catalog entry asks for.
_SEMANTIC_FINDINGS = [
    {
        "violation_text": (
            "ERS mandates rise-scheduled READ-serializer nibble presentation "
            "but the per-block qspi_slave_engine requirement advances the "
            "serializer on sck_fall."
        ),
        "evidence": (
            "A: ERS — \"Output nibbles shall be SCHEDULED DURING THE LOW PHASE "
            "FROM THE ALREADY-CONSUMED PRECEDING SYNCHRONIZED RISING-SCK "
            "EVENT\" || B: ERS per-block qspi_slave_engine — \"every sck_fall "
            "in data phase updates one nibble without a bubble\""
        ),
        "suggested_fix": "Re-issue the per-block bullet as rise-scheduled.",
        "source_doc": "ers",
    },
    {
        "violation_text": (
            "interface_contracts qspi_drive.rate_description schedules the "
            "drive controls from the falling edge; the ERS requires the "
            "already-consumed rising edge."
        ),
        "evidence": (
            "A: interface_contracts.json contracts[1].rate_description — "
            "\"Static drive controls updated by the engine only on "
            "synchronized SCK falling-edge events\" || B: ERS — \"SCHEDULED "
            "... FROM THE ALREADY-CONSUMED PRECEDING SYNCHRONIZED RISING-SCK "
            "EVENT\""
        ),
        "suggested_fix": "Re-freeze rate_description as rise-scheduled.",
        "source_doc": "block_diagram",
    },
    {
        "violation_text": (
            "interface_contracts edge_event state_semantics nibble_sampling "
            "advances the read serializer on sck_fall, contradicting the ERS."
        ),
        "evidence": (
            "A: interface_contracts.json contracts[0].representations."
            "state_semantics[0].rule — \"on sck_fall=1 it advances only the "
            "read serializer\" || B: ERS — \"advance to the low nibble after "
            "the high data sample is consumed\""
        ),
        "suggested_fix": "Rewrite nibble_sampling to advance on sck_rise.",
        "source_doc": "block_diagram",
    },
    {
        "violation_text": (
            "block_diagram qspi_drive connection semantic_contract claims "
            "stability from each synchronized falling-SCK update."
        ),
        "evidence": (
            "A: block_diagram.json connections[2].semantic_contract — "
            "\"stable from each synchronized falling-SCK update through the "
            "next host rising-edge sample\" || B: ERS — \"FROM THE "
            "ALREADY-CONSUMED PRECEDING SYNCHRONIZED RISING-SCK EVENT\""
        ),
        "suggested_fix": "Restate the semantic_contract as rise-scheduled.",
        "source_doc": "block_diagram",
    },
]


class TestCatalogEntry:

    def test_registered_and_owned(self):
        entry = next(
            c for c in cmod._CONSTRAINT_CATALOG
            if c["id"] == "cross_artifact_consistency"
        )
        assert entry["category"] == "structural"
        assert entry["severity"] == "error"
        desc = entry["description"]
        # The shared subagent prompt says "another subagent owns it"; this
        # entry must claim ownership or the finding is actively suppressed.
        assert "YOU OWN CROSS-ARTIFACT CONTRADICTIONS" in desc
        assert "does NOT apply" in desc
        # It must demand both sides of every finding.
        assert "MUST cite both sides" in desc
        assert "||" in desc

    def test_shared_prompt_still_carries_the_suppression_line(self):
        """Pins the reason the entry has to claim ownership explicitly.

        If this line is ever removed from the shared prompt, the defensive
        wording in the entry becomes redundant rather than load-bearing --
        and whoever removes it should see this test and decide deliberately.
        """
        assert (
            "another subagent owns it" in cmod._SUBAGENT_SYSTEM_PROMPT
        )

    def test_applies_needs_two_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_CROSS_ARTIFACT_GATE", raising=False)
        entry = next(
            c for c in cmod._CONSTRAINT_CATALOG
            if c["id"] == "cross_artifact_consistency"
        )
        assert entry["applies"]({}) is False
        assert entry["applies"]({
            "block_diagram": {"blocks": [{"name": "a"}]},
        }) is False
        assert entry["applies"]({
            "block_diagram": {"blocks": [{"name": "a"}]},
            "interface_contracts": {"contracts": [{"edge_id": "x"}]},
        }) is True

    def test_applies_false_when_gate_disabled(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CROSS_ARTIFACT_GATE", "0")
        entry = next(
            c for c in cmod._CONSTRAINT_CATALOG
            if c["id"] == "cross_artifact_consistency"
        )
        assert entry["applies"]({
            "block_diagram": {"blocks": [{"name": "a"}]},
            "interface_contracts": {"contracts": [{"edge_id": "x"}]},
        }) is False


class TestLlmHalfCatchesContradictions2To5:

    def _check(self, root: Path):
        bd, ic = _load(root)
        stub = _subagent_stub({
            "cross_artifact_consistency": {
                "pass": False, "violations": _SEMANTIC_FINDINGS,
            },
        })
        with patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = stub
            return _run(cmod.check_constraints(
                block_diagram=bd, memory_map={}, clock_tree={},
                register_spec={}, project_root=str(root),
                interface_contracts=ic,
            )), stub

    def test_all_four_surface_as_blocking_candidates(
        self, contradictory_project,
    ):
        violations, stub = self._check(contradictory_project)
        stub.assert_awaited()

        found = [
            v for v in violations
            if v["check"] == "cross_artifact_consistency"
            and v.get("finding_kind") == "llm_candidate"
        ]
        assert len(found) == 4, [v["violation"][:80] for v in found]

        for v in found:
            assert v["category"] == "structural"
            assert v["severity"] == "error"
            # Rendered as a candidate, not as proven fact.
            assert v["violation"].startswith("CANDIDATE CONTRADICTION (LLM):")
            # Both sides split out so a human can adjudicate.
            assert len(v["locations"]) == 2
            assert {loc["side"] for loc in v["locations"]} == {"A", "B"}
            for loc in v["locations"]:
                assert loc["artifact"] and loc["quote"]

        blob = " ".join(v["violation"] for v in found)
        assert "sck_fall" in blob                     # contradiction 2
        assert "rate_description" in blob             # contradiction 3
        assert "nibble_sampling" in blob              # contradiction 4
        assert "semantic_contract" in blob            # contradiction 5

    def test_subagent_sees_every_side_of_every_contradiction(
        self, contradictory_project,
    ):
        """The production bundle must actually contain both sides.

        A finding the subagent could never have made from its input is worth
        nothing, so assert the real ``_build_artifact_bundle`` output carries
        the contradicting text from all four artifacts.
        """
        bd, ic = _load(contradictory_project)
        bundle = cmod._build_artifact_bundle(
            block_diagram=bd, memory_map={}, clock_tree={}, register_spec={},
            benchmark_results=None, pdk_config=None, requirements="",
            ers_spec={"prd": {}, "ers": {}},
            project_root=str(contradictory_project),
        )
        # ERS both sides (contradiction 2)
        assert "PRECEDING SYNCHRONIZED RISING-SCK EVENT" in bundle
        assert "every sck_fall in data phase updates one nibble" in bundle
        # contracts (contradictions 3 + 4)
        assert "only on synchronized SCK falling-edge events" in bundle
        assert "on sck_fall=1 it advances only the read serializer" in bundle
        # block diagram (contradiction 5)
        assert "synchronized falling-SCK update" in bundle
        # FRD (contradiction 1)
        assert "12.5 MHz" in bundle

    def test_no_false_positive_when_subagent_passes(self, clean_project):
        bd, ic = _load(clean_project)
        stub = _subagent_stub({})  # every check passes
        with patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = stub
            violations = _run(cmod.check_constraints(
                block_diagram=bd, memory_map={}, clock_tree={},
                register_spec={}, project_root=str(clean_project),
                interface_contracts=ic,
            ))
        assert [
            v for v in violations if v["check"] == "cross_artifact_consistency"
        ] == []

    def test_deterministic_half_also_surfaces_through_check_constraints(
        self, contradictory_project,
    ):
        """The public entry point, not just the private helper.

        ``check_constraints`` resolves the prose artifacts from
        ``project_root`` on disk -- exactly as ``constraint_check_node``
        calls it -- so this proves the disk path, not an injected argument.
        """
        violations, _stub = self._check(contradictory_project)
        deterministic = [
            v for v in violations
            if v["check"] == "cross_artifact_consistency"
            and v.get("finding_kind") == "deterministic_quantity"
        ]
        assert deterministic, "contradiction 1 did not reach check_constraints"
        assert {v["dimension"] for v in deterministic} >= {"frequency", "time"}

    def test_unparsable_evidence_still_surfaces_with_a_warning(self):
        violations = [{
            "check": "cross_artifact_consistency",
            "violation": "two things disagree",
            "evidence": "they just do",
            "category": "structural", "severity": "error",
        }]
        cmod._annotate_cross_artifact_candidates(violations)
        v = violations[0]
        assert v["finding_kind"] == "llm_candidate"
        assert v["locations"] == []
        assert "did not cite two locations" in v["adjudication_note"]
        assert v["violation"].startswith("CANDIDATE CONTRADICTION (LLM):")


# ---------------------------------------------------------------------------
# Fail-closed wiring
# ---------------------------------------------------------------------------

class TestFailsClosed:
    """A contradiction must block Finalize, not scroll past in a log."""

    def test_production_node_path_end_to_end(self, contradictory_project):
        """constraint_check_node -> check_constraints -> route: BLOCKED.

        The whole chain with production functions only. The state is shaped
        exactly as the daemon seeds it (``ers_spec`` threaded from the ERS
        node, optional stages None); the only thing patched is the LLM.
        """
        from orchestrator.langgraph import architecture_graph as ag

        bd, _ic = _load(contradictory_project)
        state = {
            "project_root": str(contradictory_project),
            "requirements": "", "round": 1, "max_rounds": 3,
            "prd_spec": {}, "sad_spec": {}, "frd_spec": {},
            "ers_spec": {"ers": {"title": "raster2d_qspi"}},
            "block_diagram": bd,
            "memory_map": None, "clock_tree": None, "register_spec": None,
            "benchmark_data": None, "pdk_config": {},
        }
        stub = _subagent_stub({})  # every LLM check PASSES -- only the
        #                            deterministic half may fail the run.
        with patch.object(
            ag, "_persist_intermediate_state", lambda *_a, **_k: None,
        ), patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = stub
            update = _run(ag.constraint_check_node(state))

        cr = update["constraint_result"]
        assert cr["all_pass"] is False
        assert cr["has_structural"] is True
        assert any(
            v["check"] == "cross_artifact_consistency"
            for v in cr["violations"]
        )
        assert ag.route_after_constraints(
            {**state, **update},
        ) == "Escalate Constraints"

    def test_production_node_path_clean_artifacts_pass(self, clean_project):
        """The same chain on the corrected artifacts must reach Finalize."""
        from orchestrator.langgraph import architecture_graph as ag

        bd, _ic = _load(clean_project)
        state = {
            "project_root": str(clean_project),
            "requirements": "", "round": 1, "max_rounds": 3,
            "prd_spec": {}, "sad_spec": {}, "frd_spec": {},
            "ers_spec": {"ers": {"title": "raster2d_qspi"}},
            "block_diagram": bd,
            "memory_map": None, "clock_tree": None, "register_spec": None,
            "benchmark_data": None, "pdk_config": {},
        }
        stub = _subagent_stub({})
        with patch.object(
            ag, "_persist_intermediate_state", lambda *_a, **_k: None,
        ), patch(
            "orchestrator.langchain.agents.coresmith_llm.ClaudeLLM"
        ) as MockLLM:
            MockLLM.return_value.call = stub
            update = _run(ag.constraint_check_node(state))

        assert update["constraint_result"]["violations"] == []
        assert ag.route_after_constraints(
            {**state, **update},
        ) == "Finalize Architecture"

    def test_routes_to_escalation_not_finalize(self):
        from orchestrator.langgraph import architecture_graph as ag

        violations = [{
            "violation": "CROSS-ARTIFACT CONTRADICTION (frequency ...)",
            "category": "structural",
            "check": "cross_artifact_consistency",
            "severity": "error",
        }]
        state = {
            "round": 1, "max_rounds": 3,
            "constraint_result": {
                "violations": violations, "all_pass": False,
                "has_structural": True,
            },
        }
        assert ag.route_after_constraints(state) == "Escalate Constraints"

    def test_escalation_payload_carries_both_locations(self):
        from orchestrator.langgraph import architecture_graph as ag

        captured: dict = {}

        def _fake_interrupt(payload):
            captured.update(payload)
            return {"action": "accept"}

        violation = {
            "violation": "CANDIDATE CONTRADICTION (LLM): ...",
            "category": "structural",
            "check": "cross_artifact_consistency",
            "severity": "error",
            "finding_kind": "llm_candidate",
            "locations": [
                {"side": "A", "artifact": "ERS", "quote": "rise-scheduled"},
                {"side": "B", "artifact": "interface_contracts.json",
                 "quote": "falling-edge events"},
            ],
        }
        state = {
            "project_root": ".", "round": 1, "max_rounds": 3,
            "constraint_result": {
                "violations": [violation], "all_pass": False,
                "has_structural": True,
            },
            "block_diagram": {"blocks": []},
            "memory_map": None,
        }
        with patch.object(ag, "interrupt", _fake_interrupt):
            _run(ag.escalate_constraints_node(state))

        assert captured["type"] == "architecture_review_needed"
        # The operator sees the finding AND both cited sides.
        surfaced = captured["structural_violations"][0]
        assert surfaced["check"] == "cross_artifact_consistency"
        assert len(surfaced["locations"]) == 2
        # `accept` is the documented supersession path -- the gate blocks
        # Finalize until the operator explicitly takes it.
        assert "accept" in captured["supported_actions"]

    def test_accept_is_the_only_way_past(self):
        from orchestrator.langgraph import architecture_graph as ag

        base = {
            "constraint_result": {
                "violations": [{
                    "category": "structural",
                    "check": "cross_artifact_consistency",
                }],
                "has_structural": True,
            },
        }
        assert ag.route_after_constraint_escalation(
            {**base, "human_response": {"action": "accept"}},
        ) == "Finalize Architecture"
        assert ag.route_after_constraint_escalation(
            {**base, "human_response": {"action": "abort"}},
        ) == "Abort"
        # Anything else loops back into the design, never to Finalize.
        assert ag.route_after_constraint_escalation(
            {**base, "human_response": {"action": "retry"}},
        ) != "Finalize Architecture"


# ---------------------------------------------------------------------------
# Emission order -- the ERS must EXIST when Constraint Check runs
# ---------------------------------------------------------------------------

class TestErsEmissionOrder:
    """Before this change ``arch/ers_spec.md`` was written by
    ``create_documentation_node``, which runs after ``Finalize Architecture``.
    Any ERS-vs-anything check was therefore structurally impossible."""

    def test_graph_wires_ers_between_register_spec_and_constraint_check(self):
        from langgraph.checkpoint.memory import MemorySaver

        from orchestrator.langgraph.architecture_graph import (
            build_architecture_graph,
        )

        graph = build_architecture_graph(checkpointer=MemorySaver())
        assert "Engineering Requirements" in graph.get_graph().nodes
        edges = set(graph.builder.edges)
        assert ("Register Spec", "Engineering Requirements") in edges
        assert ("Engineering Requirements", "Constraint Check") in edges
        # The old direct hop must be gone, or the ERS is bypassed.
        assert ("Register Spec", "Constraint Check") not in edges
        # Doc Fix refreshes the ERS from the repaired FRD before re-checking.
        assert ("Doc Fix", "Engineering Requirements") in edges
        assert ("Doc Fix", "Constraint Check") not in edges

    def test_ers_exists_on_disk_when_constraint_check_runs(self, tmp_path):
        """Production node functions only -- nothing injected by the test.

        ``engineering_requirements_node`` writes the ERS; then the real
        ``constraint_check_node`` runs against the state IT produced, and the
        artifact bundle ``check_constraints`` builds must contain the ERS.
        """
        from orchestrator.langgraph import architecture_graph as ag

        project_root = tmp_path / "run"
        (project_root / ".coresmith").mkdir(parents=True)

        ers_payload = {"ers": {
            "title": "T", "revision": "1.0", "summary": "S",
            "functional_requirements": [
                "Legal QSPI timing is asynchronous SCK <=6.25 MHz.",
            ],
        }}

        state = {
            "project_root": str(project_root),
            "requirements": "", "round": 1, "max_rounds": 3,
            "prd_spec": {}, "sad_spec": {}, "frd_spec": {},
            "ers_spec": None,               # exactly what the daemon seeds
            "block_diagram": {"blocks": [{"name": "a"}, {"name": "b"}],
                              "connections": []},
            "memory_map": None, "clock_tree": None, "register_spec": None,
            "benchmark_data": None, "pdk_config": {},
        }

        # 1. The ERS node runs and writes the file.
        assert not (project_root / "arch" / "ers_spec.md").exists()
        with patch.object(
            ag, "_persist_intermediate_state", lambda *_a, **_k: None,
        ), patch(
            "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
            new=AsyncMock(return_value=ers_payload),
        ):
            update = _run(ag.engineering_requirements_node(state))
        assert (project_root / "arch" / "ers_spec.md").exists()
        assert update["ers_spec"] == ers_payload

        # 2. Constraint Check runs on the state that produced, and the bundle
        #    it hands the subagents contains the ERS.
        merged = {**state, **update}
        seen: dict = {}

        async def _spy_check_constraints(**kwargs):
            seen.update(kwargs)
            return []

        with patch.object(
            ag, "_persist_intermediate_state", lambda *_a, **_k: None,
        ), patch(
            "orchestrator.architecture.constraints.check_constraints",
            new=_spy_check_constraints,
        ):
            _run(ag.constraint_check_node(merged))

        assert seen["ers_spec"]["ers"] == ers_payload, (
            "constraint_check_node did not thread the ERS the ERS node emitted"
        )
        bundle = cmod._build_artifact_bundle(
            block_diagram=merged["block_diagram"], memory_map={},
            clock_tree={}, register_spec={}, benchmark_results=None,
            pdk_config=None, requirements="", ers_spec=seen["ers_spec"],
            project_root=str(project_root),
        )
        assert "ERS (Engineering Requirements Specification)" in bundle
        assert "Legal QSPI timing is asynchronous SCK <=6.25 MHz." in bundle

    def test_documentation_node_reuses_the_gated_ers(self, tmp_path):
        """No double-write, and the shipped ERS is the one the gate approved."""
        from orchestrator.langgraph import architecture_graph as ag

        project_root = tmp_path / "run"
        (project_root / ".coresmith").mkdir(parents=True)
        approved = {"ers": {"title": "APPROVED-BY-GATE", "revision": "1.0"}}

        regenerated = AsyncMock(
            return_value={"ers": {"title": "REGENERATED", "revision": "9.9"}},
        )
        state = {
            "project_root": str(project_root), "round": 1,
            "block_diagram": {"blocks": [{"name": "a"}]},
            "prd_spec": {}, "sad_spec": {}, "frd_spec": {},
            "ers_spec": approved,
            "memory_map": None, "clock_tree": None, "register_spec": None,
        }
        with patch(
            "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
            new=regenerated,
        ), patch(
            "orchestrator.architecture.specialists.dashboard_doc"
            ".generate_dashboard",
            new=AsyncMock(return_value="<html></html>"),
        ):
            result = _run(ag.create_documentation_node(state))

        regenerated.assert_not_awaited()
        assert result["ers_spec"] == approved

    def test_reorder_can_be_reverted_by_env(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import architecture_graph as ag

        monkeypatch.setenv("CORESMITH_ERS_BEFORE_CONSTRAINTS", "0")
        assert ag._ers_before_constraints_enabled() is False

        project_root = tmp_path / "run"
        (project_root / ".coresmith").mkdir(parents=True)
        never = AsyncMock(return_value={"ers": {}})
        with patch(
            "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
            new=never,
        ):
            update = _run(ag.engineering_requirements_node({
                "project_root": str(project_root), "round": 1,
                "block_diagram": {"blocks": [{"name": "a"}]},
            }))
        never.assert_not_awaited()
        assert "ers_spec" not in update
        assert not (project_root / "arch" / "ers_spec.md").exists()

    def test_ers_default_is_before_constraints(self, monkeypatch):
        from orchestrator.langgraph import architecture_graph as ag

        monkeypatch.delenv("CORESMITH_ERS_BEFORE_CONSTRAINTS", raising=False)
        assert ag._ers_before_constraints_enabled() is True

    def test_ers_failure_is_not_fatal(self, tmp_path):
        """A dead ERS generator must not take the architecture run with it."""
        from orchestrator.langgraph import architecture_graph as ag

        project_root = tmp_path / "run"
        (project_root / ".coresmith").mkdir(parents=True)
        with patch(
            "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
            new=AsyncMock(side_effect=RuntimeError("provider down")),
        ):
            update = _run(ag.engineering_requirements_node({
                "project_root": str(project_root), "round": 1,
                "block_diagram": {"blocks": [{"name": "a"}]},
                "prd_spec": {}, "sad_spec": {}, "frd_spec": {},
                "memory_map": None, "clock_tree": None, "register_spec": None,
            }))
        assert update == {"phase": "ers"}
