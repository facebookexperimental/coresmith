# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``sim.run()`` is the Amaranth simulator, not a call to the oracle.

Two consecutive runs lost their composition gate to this. The reference entry
was named ``run``, the chip model ended (as every Amaranth driver does) with::

    sim.add_testbench(bench)
    sim.run()

and the ANTI-CHEAT check matched attribute names blind::

    ANTI-CHEAT: chip model calls 'run', matching the reference entry 'run'.
    Never call the oracle from the chip model.

The model was rejected, ``_chip_model.py`` was never written, and the
composition gate -- the run's model-vs-golden check -- no-opped in silence.

The relaxation is scoped: only a method call on a name this module BOUND to a
value it constructed is spared, and only when the receiver is not itself
oracle-shaped and the module does no dynamic importing.
"""
from __future__ import annotations

import pytest

from orchestrator.langchain.agents.model_integration_generator import (
    _chip_model_attempts,
    _validate_chip_model_text,
)

_HEAD = (
    "from amaranth import Elaboratable, Module\n"
    "from amaranth.sim import Simulator\n"
    "\n"
    "class chip_model(Elaboratable):\n"
    "    def elaborate(self, platform):\n"
    "        return Module()\n"
    "\n"
)


def _model(body: str) -> str:
    return _HEAD + "def simulate(stimulus):\n" + body


class TestTheFalsePositiveThatCostTwoRuns:

    def test_the_amaranth_simulator_is_not_the_oracle(self):
        src = _model(
            "    sim = Simulator(chip_model())\n"
            "    sim.add_testbench(lambda ctx: None)\n"
            "    sim.run()\n"
            "    return [], 0\n")
        assert _validate_chip_model_text(src, "run") is None

    def test_a_bare_call_to_the_entry_is_still_a_cheat(self):
        src = _model("    return run(stimulus), 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_an_oracle_shaped_receiver_is_still_a_cheat(self):
        """Even locally bound. ``golden = load(...)`` then ``golden.run()`` is
        exactly the shape the check exists to stop."""
        for recv in ("golden", "reference", "oracle", "ref", "_golden"):
            src = _model(f"    {recv} = object()\n"
                         f"    return {recv}.run(stimulus), 0\n")
            assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or ""), recv

    def test_an_unbound_receiver_is_still_a_cheat(self):
        """``mod.run()`` where nothing in the module ever bound ``mod`` is a
        module-level handle -- the way an oracle arrives."""
        src = _model("    return mod.run(stimulus), 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_a_chained_receiver_is_still_a_cheat(self):
        src = _model("    return pkg.mod.run(stimulus), 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_dynamic_import_restores_the_strict_rule(self):
        """A module that can bind the oracle to a local at runtime does not get
        the local-object exception."""
        src = _model(
            "    import importlib\n"
            "    sim = importlib.import_module('x')\n"
            "    sim.run()\n"
            "    return [], 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    @pytest.mark.parametrize("recv_binding", [
        "    sim = Simulator(chip_model())\n",
        "    for sim in sims:\n        pass\n",
        "    with make() as sim:\n        pass\n",
    ])
    def test_every_local_binding_form_counts(self, recv_binding):
        src = _model(recv_binding + "    sim.run()\n    return [], 0\n")
        assert _validate_chip_model_text(src, "run") is None, recv_binding


class TestTheOriginalChecksAreIntact:

    def test_importing_the_golden_is_rejected(self):
        src = "import design_golden\n" + _model("    return [], 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_from_importing_the_golden_is_rejected(self):
        src = "from design_golden import run\n" + _model("    return [], 0\n")
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_reimplementing_the_entry_is_rejected(self):
        src = _model("    return [], 0\n") + "\ndef _run(stimulus):\n    return []\n"
        assert "ANTI-CHEAT" in (_validate_chip_model_text(src, "run") or "")

    def test_structural_requirements_still_apply(self):
        assert "missing module-level def simulate" in (
            _validate_chip_model_text(_HEAD, "run") or "")
        assert "missing class chip_model" in (
            _validate_chip_model_text("def simulate(s):\n    return [], 0\n",
                                      "run") or "")

    def test_no_entry_name_means_no_entry_check(self):
        src = _model("    return run(stimulus), 0\n")
        assert _validate_chip_model_text(src, "") is None


class TestTheRetryBudget:

    def test_default_is_two(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_CHIP_MODEL_ATTEMPTS", raising=False)
        assert _chip_model_attempts() == 2

    def test_single_shot_is_recoverable(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CHIP_MODEL_ATTEMPTS", "1")
        assert _chip_model_attempts() == 1

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CHIP_MODEL_ATTEMPTS", "nonsense")
        assert _chip_model_attempts() == 2


class TestTheNoOpIsLoud:

    @pytest.mark.asyncio
    async def test_a_failed_chip_model_is_carried_forward(self, tmp_path,
                                                          monkeypatch):
        """The composition gate silently losing its input is the single most
        expensive silent outcome in the flow, so it lands in the ledger the
        final report and validation DV both read."""
        from orchestrator.architecture import composition as _composition
        from orchestrator.langgraph import pipeline_graph as pg

        models = tmp_path / "arch" / _composition.BLOCK_MODELS_DIRNAME
        models.mkdir(parents=True)
        (models / "b.py").write_text("# a block model\n")
        (tmp_path / "inputs").mkdir()
        ref = tmp_path / "inputs" / "design_golden.py"
        ref.write_text("def run(stimulus):\n    return []\n")

        monkeypatch.setattr(_composition, "resolve_generator_reference",
                            lambda pr: str(ref))

        class _Boom:
            def __init__(self, *a, **k):
                pass

            async def generate(self, **kw):
                raise RuntimeError(
                    "Integrated chip model at x is invalid: ANTI-CHEAT: chip "
                    "model calls 'run'")

        import orchestrator.langchain.agents.model_integration_generator as mig
        monkeypatch.setattr(mig, "ModelIntegrationGenerator", _Boom)

        await pg._maybe_generate_chip_model(str(tmp_path))

        defects = pg.read_carried_forward_defects(str(tmp_path))
        assert len(defects) == 1
        d = defects[0]
        assert d["gate"] == "model_integration"
        assert d["kind"] == "chip_model_generation_failed"
        assert "composition gate" in d["detail"]
        assert "ANTI-CHEAT" in d["detail"]


class TestTheRetryCarriesTheRejectionBack:
    """The rules were in the prompt the first time. What was missing was the
    validator's complaint, which localises the fault to one line."""

    @staticmethod
    def _agent(responses):
        from orchestrator.langchain.agents.model_integration_generator import (
            ModelIntegrationGenerator,
        )
        seen = []

        class _LLM:
            async def call(self, system, prompt, run_name=""):
                seen.append(prompt)
                return responses[min(len(seen) - 1, len(responses) - 1)]

        # Bypass __init__: constructing ClaudeLLM needs the CLI on PATH, which
        # has nothing to do with what this asserts.
        agent = object.__new__(ModelIntegrationGenerator)
        agent.llm = _LLM()
        return agent, seen

    @staticmethod
    def _fenced(body):
        return "```python\n" + _model(body) + "```\n"

    @pytest.mark.asyncio
    async def test_a_rejected_model_is_regenerated_with_the_reason(self,
                                                                   tmp_path):
        agent, seen = self._agent([
            self._fenced("    return run(stimulus), 0\n"),      # cheats
            self._fenced("    sim = Simulator(chip_model())\n"
                         "    sim.run()\n    return [], 0\n"),  # clean
        ])
        models = tmp_path / "models"
        models.mkdir()
        (models / "b.py").write_text("# block model\n")
        out = models / "_chip_model.py"
        res = await agent.generate(
            project_root=str(tmp_path), block_models_dir=str(models),
            block_diagram={}, interface_contracts={},
            reference_impl_source="", reference_entry_name="run",
            output_path=str(out))
        assert res["path"] == str(out)
        assert len(seen) == 2
        assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in seen[1]
        assert "ANTI-CHEAT" in seen[1]
        assert "sim.run()" in out.read_text()

    @pytest.mark.asyncio
    async def test_a_clean_first_attempt_costs_nothing_extra(self, tmp_path):
        agent, seen = self._agent([
            self._fenced("    sim = Simulator(chip_model())\n"
                         "    sim.run()\n    return [], 0\n"),
        ])
        models = tmp_path / "models"
        models.mkdir()
        await agent.generate(
            project_root=str(tmp_path), block_models_dir=str(models),
            block_diagram={}, interface_contracts={},
            reference_impl_source="", reference_entry_name="run",
            output_path=str(models / "_chip_model.py"))
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_two_bad_attempts_still_raise(self, tmp_path):
        agent, seen = self._agent([self._fenced("    return run(stimulus), 0\n")])
        models = tmp_path / "models"
        models.mkdir()
        with pytest.raises(RuntimeError, match="ANTI-CHEAT"):
            await agent.generate(
                project_root=str(tmp_path), block_models_dir=str(models),
                block_diagram={}, interface_contracts={},
                reference_impl_source="", reference_entry_name="run",
                output_path=str(models / "_chip_model.py"))
        assert len(seen) == 2
