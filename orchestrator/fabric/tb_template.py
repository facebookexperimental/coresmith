# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The cocotb testbench generated for every fabric instance (cocotbext-axi)."""
from __future__ import annotations

import json

from .spec import FabricSpec

_TB = '''"""Generated fabric testbench for {module} (B1) -- do not edit.

Drives every master port with a cocotbext-axi AxiMaster, models every slave
(AxiRam / AxiLiteRam / a small APB memory) and checks: address decode,
decode errors on unmapped addresses, bursts, per-master ordering, random
backpressure and fairness under contention.
"""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, ReadOnly, Timer

from cocotbext.axi import AxiBus, AxiLiteBus, AxiLiteRam, AxiMaster, AxiRam, AxiResp

MASTERS = {masters}
SLAVES = {slaves}          # name -> (protocol, base, size)
DW = {dw}
UNMAPPED = {unmapped}


class ApbMem:
    """A tiny APB completer: one-cycle pready, byte-addressed memory."""

    def __init__(self, dut, prefix, size):
        self.dut, self.p, self.size = dut, prefix, size
        self.mem = bytearray(size)

    def _s(self, n):
        return getattr(self.dut, f"{{self.p}}_{{n}}")

    async def run(self):
        self._s("pready").value = 0
        self._s("pslverr").value = 0
        self._s("prdata").value = 0
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            sel = int(self._s("psel").value) and int(self._s("penable").value)
            await Timer(1, "step")
            if sel:
                addr = int(self._s("paddr").value) % self.size
                if int(self._s("pwrite").value):
                    data = int(self._s("pwdata").value)
                    strb = int(self._s("pstrb").value)
                    for i in range(DW // 8):
                        if (strb >> i) & 1:
                            self.mem[(addr + i) % self.size] = (data >> (8 * i)) & 0xFF
                    self._s("pready").value = 1
                else:
                    val = 0
                    for i in range(DW // 8):
                        val |= self.mem[(addr + i) % self.size] << (8 * i)
                    self._s("prdata").value = val
                    self._s("pready").value = 1
            else:
                self._s("pready").value = 0


class Env:
    def __init__(self, dut):
        self.dut = dut
        cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
        self.masters = {{}}
        self.slaves = {{}}
        for m in MASTERS:
            self.masters[m] = AxiMaster(AxiBus.from_prefix(dut, f"s_{{m}}"), dut.clk, dut.rst_n,
                                        reset_active_level=False)
        for s, (proto, base, size) in SLAVES.items():
            if proto == "axi4":
                self.slaves[s] = AxiRam(AxiBus.from_prefix(dut, f"m_{{s}}"), dut.clk, dut.rst_n,
                                        reset_active_level=False, size=size)
            elif proto == "axi_lite":
                self.slaves[s] = AxiLiteRam(AxiLiteBus.from_prefix(dut, f"m_{{s}}"), dut.clk, dut.rst_n,
                                            reset_active_level=False, size=size)
            else:
                mem = ApbMem(dut, f"m_{{s}}", size)
                cocotb.start_soon(mem.run())
                self.slaves[s] = mem

    async def reset(self):
        self.dut.rst_n.value = 0
        await ClockCycles(self.dut.clk, 5)
        self.dut.rst_n.value = 1
        await ClockCycles(self.dut.clk, 5)

    def set_backpressure(self, on):
        gen = (lambda: (random.random() < 0.4)) if on else None
        for s in self.slaves.values():
            for attr in ("write_if", "read_if"):
                itf = getattr(s, attr, None)
                if itf is None:
                    continue
                for ch in ("aw_channel", "w_channel", "b_channel", "ar_channel", "r_channel"):
                    c = getattr(itf, ch, None)
                    if c is not None and hasattr(c, "set_pause_generator"):
                        c.set_pause_generator(_gen(gen) if gen else None)


def _gen(fn):
    while True:
        yield fn()


def _word(seed):
    rng = random.Random(repr(seed))
    return bytes(rng.getrandbits(8) for _ in range(DW // 8))


@cocotb.test()
async def test_decode(dut):
    """Every master reaches every slave at its base and its last word."""
    env = Env(dut)
    await env.reset()
    for mi, m in enumerate(MASTERS):
        for s, (proto, base, size) in SLAVES.items():
            for addr in (base, base + size - DW // 8):
                data = _word((mi, s, addr))
                await env.masters[m].write(addr, data)
                rd = await env.masters[m].read(addr, len(data))
                assert rd.resp == AxiResp.OKAY, f"{{m}}->{{s}} @{{addr:#x}}: resp {{rd.resp}}"
                assert bytes(rd.data) == data, f"{{m}}->{{s}} @{{addr:#x}}: {{bytes(rd.data).hex()}} != {{data.hex()}}"


@cocotb.test()
async def test_decode_error(dut):
    """An unmapped address answers DECERR and does not wedge the fabric."""
    env = Env(dut)
    await env.reset()
    m = env.masters[MASTERS[0]]
    rd = await m.read(UNMAPPED, DW // 8)
    assert rd.resp == AxiResp.DECERR, f"unmapped read resp {{rd.resp}}"
    wr = await m.write(UNMAPPED, bytes(DW // 8))
    assert wr.resp == AxiResp.DECERR, f"unmapped write resp {{wr.resp}}"
    s, (proto, base, size) = next(iter(SLAVES.items()))
    data = _word(("after", s))
    await m.write(base, data)
    rd = await m.read(base, len(data))
    assert bytes(rd.data) == data


@cocotb.test()
async def test_bursts(dut):
    """A 64-byte burst write/read round-trips through an AXI4 slave."""
    env = Env(dut)
    await env.reset()
    axi4 = [(s, v) for s, v in SLAVES.items() if v[0] == "axi4"]
    if not axi4:
        return
    s, (proto, base, size) = axi4[0]
    m = env.masters[MASTERS[0]]
    data = bytes(random.getrandbits(8) for _ in range(min(64, size)))
    await m.write(base, data)
    rd = await m.read(base, len(data))
    assert bytes(rd.data) == data


@cocotb.test()
async def test_backpressure_and_ordering(dut):
    """Random slave stalls: responses stay correct and in order per master."""
    env = Env(dut)
    await env.reset()
    env.set_backpressure(True)
    s, (proto, base, size) = next(iter(SLAVES.items()))
    m = env.masters[MASTERS[-1]]
    words = [_word(("bp", i)) for i in range(8)]
    for i, w in enumerate(words):
        await m.write(base + i * (DW // 8), w)
    reads = [await m.read(base + i * (DW // 8), DW // 8) for i in range(8)]
    env.set_backpressure(False)
    assert [bytes(r.data) for r in reads] == words


@cocotb.test()
async def test_fairness(dut):
    """Every master completes its share while all hammer one slave."""
    env = Env(dut)
    await env.reset()
    s, (proto, base, size) = next(iter(SLAVES.items()))
    n = 6

    async def hammer(mi, m):
        for i in range(n):
            addr = base + ((mi * n + i) * (DW // 8)) % max(DW // 8, size - DW // 8)
            data = _word(("fair", mi, i))
            await env.masters[m].write(addr, data)
            rd = await env.masters[m].read(addr, len(data))
            assert bytes(rd.data) == data, f"{{m}} #{{i}}"
        return mi
    tasks = [cocotb.start_soon(hammer(mi, m)) for mi, m in enumerate(MASTERS)]
    done = []
    for t in tasks:
        done.append(await t)
    assert sorted(done) == list(range(len(MASTERS)))
'''


def render_testbench(spec: FabricSpec) -> str:
    slaves = {s.name: (s.protocol, s.base, s.size) for s in spec.slaves}
    top = max(s.base + s.size for s in spec.slaves)
    unmapped = top + 0x1000 if top + 0x1000 < (1 << spec.addr_width) else 0
    # pick an unmapped address not inside any slave
    while any(s.base <= unmapped < s.base + s.size for s in spec.slaves):
        unmapped += 0x1000
    return _TB.format(module=spec.module_name, masters=json.dumps([m.name for m in spec.masters]),
                      slaves=repr(slaves), dw=spec.data_width, unmapped=hex(unmapped))
