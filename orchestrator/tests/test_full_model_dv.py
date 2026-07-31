# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Full Model DV (dv-hardening-14): uarch-stage tier 2, mission-scale
model-vs-golden on the FRD acceptance stimulus.

The armC lesson in miniature: a chip model that is CORRECT on the fast gate's
small stimulus but WRONG at scale (the cascade class) must be caught by the
acceptance tier -- the fast gate alone certified 30dB on one macroblock while
real frames landed at 21dB.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.architecture import model_integration

REFERENCE_IMPL = '''\
def run(stim):
    return [v + 1 for v in stim]
'''

# Correct on short stimuli (the gate default is 4 beats), WRONG beyond 8 beats
# -- the scale-dependent divergence class.
SCALE_BROKEN_CHIP = '''\
def simulate(stimulus):
    out = [v + 1 for v in stimulus]
    if len(stimulus) > 8:
        out[9] = 0  # scale-dependent corruption
    return out, len(stimulus)
'''

SCALE_CORRECT_CHIP = '''\
def simulate(stimulus):
    return [v + 1 for v in stimulus], len(stimulus)
'''

ACCEPTANCE_ART = '''\
stimulus = list(range(32))
'''

ACCEPTANCE_CASES_ART = '''\
cases = [("small", [1, 2, 3]), ("mission_scale", list(range(32)))]
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(tmp_path: Path, chip_text: str) -> Path:
    _write(tmp_path / "arch" / "block_models" / "blk.py", "# block model\n")
    _write(tmp_path / "arch" / "block_models" / "_chip_model.py", chip_text)
    _write(tmp_path / "inputs" / "toy_golden.py", REFERENCE_IMPL)
    return tmp_path


def _env(monkeypatch):
    monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
    monkeypatch.setenv("CORESMITH_COMPOSITION_AUDIT", "1")
    for var in ("CORESMITH_REFERENCE_ENTRY", "CORESMITH_MODEL_STIMULUS",
                "CORESMITH_FUNCTIONAL_ACCEPTANCE", "CORESMITH_SIM_PYTHON",
                "CORESMITH_ACCEPTANCE_STIMULUS", "CORESMITH_FULL_MODEL_DV",
                "CORESMITH_FIDELITY_GATE", "CORESMITH_FIDELITY_METRIC"):
        monkeypatch.delenv(var, raising=False)


class TestFullModelDV:
    def test_no_acceptance_artifact_honest_skip(self, tmp_path, monkeypatch):
        # Scale-broken chip + NO acceptance artifact: the fast gate passes on
        # its 4-beat default stimulus and full-model-dv skips honestly --
        # documenting exactly the pre-dv-14 blind spot.
        _env(monkeypatch)
        root = _project(tmp_path, SCALE_BROKEN_CHIP)
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations == [], violations

    def test_acceptance_artifact_catches_scale_divergence(
        self, tmp_path, monkeypatch
    ):
        _env(monkeypatch)
        root = _project(tmp_path, SCALE_BROKEN_CHIP)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE_ART)
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations, "scale-dependent divergence must be caught"
        v = violations[0]
        assert v["criterion"] == "full_model_dv_divergence"
        assert v["stimulus_tier"] == "acceptance"
        assert v["gap_class"] == "block_math"

    def test_scale_correct_chip_passes_acceptance(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root = _project(tmp_path, SCALE_CORRECT_CHIP)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE_ART)
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations == [], violations

    def test_cases_list_form(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root = _project(tmp_path, SCALE_BROKEN_CHIP)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE_CASES_ART)
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations
        assert violations[0]["acceptance_case"] == "mission_scale"

    def test_env_artifact_override(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root = _project(tmp_path, SCALE_BROKEN_CHIP)
        art = tmp_path / "elsewhere.py"
        _write(art, ACCEPTANCE_ART)
        monkeypatch.setenv("CORESMITH_ACCEPTANCE_STIMULUS", str(art))
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations and violations[0]["criterion"] == "full_model_dv_divergence"

    def test_kill_switch(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        monkeypatch.setenv("CORESMITH_FULL_MODEL_DV", "0")
        root = _project(tmp_path, SCALE_BROKEN_CHIP)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE_ART)
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations == []


class TestByteShiftDetection:
    """dv-hardening-18: a shifted-but-content-exact stream must be named as
    the FIFO/alignment class, not routed as unlocalized content math (armD
    live: dup head byte -> 'unlocalized block_math' -> the driver had to
    localize by hand)."""

    def test_dup_head_detected(self):
        from orchestrator.architecture.model_integration import detect_byte_shift

        e = bytes(range(40))
        o = b"\x41" + e[:-1]
        msg = detect_byte_shift(e, o)
        assert "BYTE-SHIFT DETECTED" in msg
        assert "+1" in msg and "head" in msg

    def test_dropped_head_detected(self):
        from orchestrator.architecture.model_integration import detect_byte_shift

        e = bytes(range(40))
        o = e[2:]
        msg = detect_byte_shift(e, o)
        assert "BYTE-SHIFT DETECTED" in msg and "DROPPED" in msg

    def test_no_shift_no_match(self):
        from orchestrator.architecture.model_integration import detect_byte_shift

        e = bytes(range(40))
        assert detect_byte_shift(e, e) == ""
        assert detect_byte_shift(e, bytes(reversed(e))) == ""

    def test_gate_violation_carries_shift_message(self, tmp_path, monkeypatch):
        from orchestrator.architecture import model_integration

        _env(monkeypatch)
        # chip returns golden's bytes with a duplicated head byte
        chip = '''
def simulate(stimulus):
    good = bytes((v + 1) & 0xFF for v in stimulus)
    return bytes([good[0]]) + good[:-1], len(stimulus)
'''
        root = _project(tmp_path, chip)
        # stimulus long enough for shift detection (>=8 bytes)
        art = tmp_path / "stim.py"
        art.write_text("stimulus = list(range(32))\n")
        monkeypatch.setenv("CORESMITH_MODEL_STIMULUS", str(art))
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations
        assert "BYTE-SHIFT DETECTED" in str(violations[0].get("suggested_fix", ""))


# ---------------------------------------------------------------------------
# Timeout localization (run3-followups #4b)
# ---------------------------------------------------------------------------
#
# When the composed model runs unbounded there is no divergence offset and no
# traceback, so the failure reported "First-divergence block: (unlocalized)" and
# the router broadcast a re-spec to EVERY block. A stall has one first cause, and
# this is the failure mode where saying WHERE matters most.
#
# The probe is static (ast over the composition + a standalone import of each
# block model), so it costs nothing and works on a run that is already hung.

HANGING_CHIP = '''\
def simulate(stimulus):
    while True:
        pass
'''

# A composition where one block's ports are wired to signals nothing else in the
# whole chip model touches -- an output no one consumes / an input no one drives.
DANGLING_CHIP = '''\
from amaranth import Module, Signal

from producer import producer
from orphan import orphan


class chip_model:
    def elaborate(self, platform):
        m = Module()
        shared = Signal(8)
        shared_valid = Signal()
        never_read_a = Signal(8)
        never_read_b = Signal()
        m.submodules.p = producer(shared, shared_valid)
        m.submodules.o = orphan(never_read_a, never_read_b)
        m.d.comb += shared.eq(shared_valid)
        return m


def simulate(stimulus):
    while True:
        pass
'''

BROKEN_BLOCK = "import a_module_that_does_not_exist_anywhere\n"


def _good_block(name: str) -> str:
    """A block model that imports cleanly and exposes its own name."""
    return f"class {name}:\n    def __init__(self, *args):\n        pass\n"


def _stall_project(tmp_path: Path, chip: str, blocks: dict) -> Path:
    d = tmp_path / "arch" / "block_models"
    for stem, text in blocks.items():
        _write(d / f"{stem}.py", text if text else _good_block(stem))
    _write(d / "_chip_model.py", chip)
    _write(tmp_path / "inputs" / "toy_golden.py", REFERENCE_IMPL)
    return tmp_path


class TestStallLocalization:
    def test_probe_names_a_block_that_does_not_even_import(self, tmp_path):
        root = _stall_project(tmp_path, HANGING_CHIP, {
            "producer": "", "orphan": BROKEN_BLOCK})
        rep = model_integration.probe_composition_stall(
            str(root), root / "arch" / "block_models")
        assert set(rep["blocks"]) == {"producer", "orphan"}
        assert "orphan" in rep["import_failures"]
        assert "producer" not in rep["import_failures"]
        assert rep["candidates"] == ["orphan"]
        assert "DOES NOT IMPORT" in rep["summary"]

    def test_probe_names_a_block_whose_ports_are_wired_to_nothing(self, tmp_path):
        root = _stall_project(tmp_path, DANGLING_CHIP, {
            "producer": "", "orphan": ""})
        rep = model_integration.probe_composition_stall(
            str(root), root / "arch" / "block_models")
        assert rep["dangling"].get("orphan") == ["never_read_a", "never_read_b"]
        assert "producer" not in rep["dangling"]     # its signals ARE consumed
        assert rep["candidates"] == ["orphan"]
        assert "read NOWHERE else" in rep["summary"]

    def test_probe_reports_nothing_when_the_composition_is_well_formed(
            self, tmp_path):
        chip = DANGLING_CHIP.replace(
            "m.submodules.o = orphan(never_read_a, never_read_b)",
            "m.submodules.o = orphan(shared, shared_valid)")
        root = _stall_project(tmp_path, chip, {
            "producer": "", "orphan": ""})
        rep = model_integration.probe_composition_stall(
            str(root), root / "arch" / "block_models")
        assert rep["candidates"] == []
        assert rep["summary"] == ""

    def test_probe_never_raises_on_a_missing_or_unparseable_dir(self, tmp_path):
        rep = model_integration.probe_composition_stall(
            str(tmp_path), tmp_path / "nope")
        assert rep["candidates"] == [] and rep["blocks"] == []

    def test_timeout_violation_says_WHERE(self, tmp_path, monkeypatch):
        """PRODUCTION PATH: the real gate, a real (fast) timeout, and the
        violation the router consumes now NAMES the block instead of reporting
        '(unlocalized)'."""
        _env(monkeypatch)
        monkeypatch.setenv("CORESMITH_GATE_SIM_TIMEOUT", "1")
        root = _stall_project(tmp_path, DANGLING_CHIP, {
            "producer": "", "orphan": ""})
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations, "a hanging simulate() must be a violation"
        v = violations[0]
        assert v["criterion"] == "simulate_timeout"
        assert v["first_divergence_block"] == "orphan"
        assert "LOCALIZED to block 'orphan'" in v["observed"]
        assert "read NOWHERE else" in v["suggested_fix"]
        assert v["stall_localization"]["candidates"] == ["orphan"]

    def test_localization_can_be_switched_off(self, tmp_path, monkeypatch):
        """Env-gate convention: 0 restores the pre-fix '(unlocalized)' report,
        so both branches are covered."""
        _env(monkeypatch)
        monkeypatch.setenv("CORESMITH_GATE_SIM_TIMEOUT", "1")
        monkeypatch.setenv("CORESMITH_GATE_STALL_LOCALIZE", "0")
        root = _stall_project(tmp_path, DANGLING_CHIP, {
            "producer": "", "orphan": ""})
        violations = model_integration.run_model_integration_gate(str(root))
        assert violations
        v = violations[0]
        assert v["first_divergence_block"] == ""
        assert "NOT localized" in v["observed"]
        assert v["stall_localization"] == {}

    def test_ambiguous_evidence_does_not_invent_a_single_block(self, tmp_path):
        """Two candidate blocks -> the probe reports both and names NEITHER as
        the first-divergence block. A false-precise pointer misdirects a re-spec,
        which is why the broadcast default exists."""
        root = _stall_project(tmp_path, DANGLING_CHIP, {
            "producer": BROKEN_BLOCK, "orphan": BROKEN_BLOCK})
        rep = model_integration.probe_composition_stall(
            str(root), root / "arch" / "block_models")
        assert len(rep["candidates"]) == 2
