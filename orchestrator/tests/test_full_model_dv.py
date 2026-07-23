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
