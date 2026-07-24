"""Focused contract tests for the experimental Amaranth behavioral arm."""

import shutil

import pytest

from orchestrator.architecture.composition_audit import audit_chip_model
from orchestrator.langgraph.microarch_exp import elaborate_block_model

BLOCK = '''
from amaranth import Elaboratable, Module

class inc(Elaboratable):
    def __init__(self, clk, rst, din, vin, dout, vout):
        self.clk, self.rst = clk, rst
        self.din, self.vin, self.dout, self.vout = din, vin, dout, vout

    def elaborate(self, platform):
        m = Module()
        with m.If(self.rst):
            m.d.sync += [self.dout.eq(0), self.vout.eq(0)]
        with m.Else():
            m.d.sync += self.vout.eq(self.vin)
            with m.If(self.vin):
                m.d.sync += self.dout.eq(self.din + 1)
        return m
'''


CHIP = '''
from amaranth import Elaboratable, Module, Signal
from inc import inc

class chip_model(Elaboratable):
    def __init__(self, clk, rst, din, vin, dout, vout):
        self.clk, self.rst = clk, rst
        self.din, self.vin, self.dout, self.vout = din, vin, dout, vout

    def elaborate(self, platform):
        m = Module()
        mid = Signal(8)
        mid_v = Signal()
        m.submodules.a = inc(self.clk, self.rst, self.din, self.vin, mid, mid_v)
        m.submodules.b = inc(self.clk, self.rst, mid, mid_v, self.dout, self.vout)
        return m

def simulate(stimulus):
    return [], 0
'''


@pytest.mark.skipif(
    shutil.which("yosys") is None, reason="yosys binary not available"
)
def test_elaborate_and_system_yosys_codegen(tmp_path, monkeypatch):
    model = tmp_path / "inc.py"
    model.write_text(BLOCK, encoding="utf-8")
    monkeypatch.setenv("AMARANTH_USE_YOSYS", "system")
    assert elaborate_block_model(str(model), "inc") is None
    assert (tmp_path / "_uarch_exp_elab" / "inc.v").is_file()


def test_amaranth_composition_audit_understands_constructor_and_eq(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "inc.py").write_text(BLOCK, encoding="utf-8")
    chip = models / "_chip_model.py"
    chip.write_text(CHIP, encoding="utf-8")
    result = audit_chip_model(chip, models)
    assert result.violations == []


def test_async_tick_samples_post_edge_and_narrow_signal_wraps():
    from amaranth import Elaboratable, Module, Signal
    from amaranth.sim import Simulator

    class Counter(Elaboratable):
        def __init__(self):
            self.count = Signal(4, init=15)

        def elaborate(self, platform):
            m = Module()
            m.d.sync += self.count.eq(self.count + 1)
            return m

    dut = Counter()
    sim = Simulator(dut)
    sim.add_clock(20e-9)
    seen = []

    async def bench(ctx):
        await ctx.tick()
        seen.append(ctx.get(dut.count))

    sim.add_testbench(bench)
    sim.run()
    assert seen == [0]  # post-edge value; 4-bit destination silently wraps
