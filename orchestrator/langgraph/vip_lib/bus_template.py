# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""cocotb template for the memory-mapped bus families (axi4 / axi_lite / apb),
appended verbatim (no .format) after the common VIP header (B1)."""

BUS_TEMPLATE = '''

class Driver:
    """Memory-mapped bus VIP over cocotbext-axi.

    * consumer role (the DUT is the SLAVE of this edge): ``master()`` returns a
      cocotbext ``AxiMaster`` / ``AxiLiteMaster`` (APB: a small ``ApbMaster``)
      bound to the DUT's ``<channel>_*`` ports -- issue ``read``/``write``;
    * producer role (the DUT is the MASTER): ``responder(size)`` binds a
      cocotbext ``AxiRam`` / ``AxiLiteRam`` (APB: ``ApbMem``) to the DUT's ports.
    """

    def __init__(self, dut, side):
        self.dut, self.side = dut, side
        self._obj = None

    async def reset(self):
        await _cycle_start(self.dut)

    def _prefix(self):
        return self.side["channel"]

    def master(self):
        if self._obj is None:
            if FAMILY == "axi4":
                from cocotbext.axi import AxiBus, AxiMaster
                self._obj = AxiMaster(AxiBus.from_prefix(self.dut, self._prefix()), _clk(self.dut),
                                      getattr(self.dut, RST), reset_active_level=False)
            elif FAMILY == "axi_lite":
                from cocotbext.axi import AxiLiteBus, AxiLiteMaster
                self._obj = AxiLiteMaster(AxiLiteBus.from_prefix(self.dut, self._prefix()), _clk(self.dut),
                                          getattr(self.dut, RST), reset_active_level=False)
            else:
                self._obj = ApbMaster(self.dut, self._prefix())
        return self._obj

    def responder(self, size=4096):
        if self._obj is None:
            if FAMILY == "axi4":
                from cocotbext.axi import AxiBus, AxiRam
                self._obj = AxiRam(AxiBus.from_prefix(self.dut, self._prefix()), _clk(self.dut),
                                   getattr(self.dut, RST), reset_active_level=False, size=size)
            elif FAMILY == "axi_lite":
                from cocotbext.axi import AxiLiteBus, AxiLiteRam
                self._obj = AxiLiteRam(AxiLiteBus.from_prefix(self.dut, self._prefix()), _clk(self.dut),
                                       getattr(self.dut, RST), reset_active_level=False, size=size)
            else:
                self._obj = ApbMem(self.dut, self._prefix(), size)
                cocotb.start_soon(self._obj.run())
        return self._obj


class ApbMaster:
    """A minimal APB requester (one transfer at a time)."""

    def __init__(self, dut, prefix):
        self.dut, self.p = dut, prefix

    def _s(self, n):
        return getattr(self.dut, f"{self.p}_{n}")

    async def _xfer(self, addr, write, data=0, strb=None):
        await _cycle_start(self.dut)
        self._s("psel").value = 1
        self._s("penable").value = 0
        self._s("pwrite").value = int(write)
        self._s("paddr").value = int(addr)
        if write:
            self._s("pwdata").value = int(data)
            if hasattr(self.dut, f"{self.p}_pstrb"):
                self._s("pstrb").value = strb if strb is not None else (1 << (len(self._s("pwdata")) // 8)) - 1
        await _cycle_start(self.dut)
        self._s("penable").value = 1
        while True:
            await _sample(self.dut)
            if int(self._s("pready").value):
                rdata = int(self._s("prdata").value) if not write else None
                err = int(self._s("pslverr").value) if hasattr(self.dut, f"{self.p}_pslverr") else 0
                break
            await RisingEdge(_clk(self.dut))
        await _cycle_start(self.dut)
        self._s("psel").value = 0
        self._s("penable").value = 0
        return rdata, err

    async def write(self, addr, data, strb=None):
        return (await self._xfer(addr, True, data, strb))[1]

    async def read(self, addr):
        return await self._xfer(addr, False)


class ApbMem:
    """A tiny APB completer with byte-addressed memory."""

    def __init__(self, dut, prefix, size):
        self.dut, self.p, self.size = dut, prefix, size
        self.mem = bytearray(size)

    def _s(self, n):
        return getattr(self.dut, f"{self.p}_{n}")

    async def run(self):
        self._s("pready").value = 0
        if hasattr(self.dut, f"{self.p}_pslverr"):
            self._s("pslverr").value = 0
        self._s("prdata").value = 0
        nbytes = len(self._s("pwdata")) // 8
        while True:
            await _sample(self.dut)
            sel = int(self._s("psel").value) and int(self._s("penable").value)
            await _cycle_start(self.dut)
            if sel:
                addr = int(self._s("paddr").value) % self.size
                if int(self._s("pwrite").value):
                    data = int(self._s("pwdata").value)
                    strb = int(self._s("pstrb").value) if hasattr(self.dut, f"{self.p}_pstrb") else (1 << nbytes) - 1
                    for i in range(nbytes):
                        if (strb >> i) & 1:
                            self.mem[(addr + i) % self.size] = (data >> (8 * i)) & 0xFF
                else:
                    val = 0
                    for i in range(nbytes):
                        val |= self.mem[(addr + i) % self.size] << (8 * i)
                    self._s("prdata").value = val
                self._s("pready").value = 1
            else:
                self._s("pready").value = 0


class Monitor:
    """Counts accepted beats per channel (valid && ready mid-cycle)."""

    def __init__(self, dut, side, scoreboard=None):
        self.dut, self.side, self.sb = dut, side, scoreboard
        self.counts = {ch: 0 for ch, _ in HANDSHAKES}
        self._task = None

    def start(self):
        self._task = cocotb.start_soon(self._run())
        return self

    async def _run(self):
        dut, side = self.dut, self.side
        while True:
            await _sample(dut)
            if _in_reset(dut):
                continue
            for v, r in HANDSHAKES:
                if _has(dut, side, v) and _has(dut, side, r):
                    if _val(_sig(dut, side, v)) == 1 and _val(_sig(dut, side, r)) == 1:
                        self.counts[v] += 1


async def assertions(dut, side):
    """Per-channel AMBA rules: valid holds until ready, and every valid is low
    for reset_idle_cycles after reset."""
    idle = int(TIMING.get("reset_idle_cycles") or 0)
    prev = {v: (0, 0) for v, _ in HANDSHAKES}
    since_reset = None
    while True:
        await _sample(dut)
        if _in_reset(dut):
            since_reset = 0
            prev = {v: (0, 0) for v, _ in HANDSHAKES}
            continue
        for v, r in HANDSHAKES:
            if not (_has(dut, side, v) and _has(dut, side, r)):
                continue
            cv = _val(_sig(dut, side, v)) or 0
            cr = _val(_sig(dut, side, r)) or 0
            if since_reset is not None and since_reset < idle and cv:
                raise VIPError(f"{EDGE_ID}: {v} high {since_reset} cycle(s) after reset (reset_idle_cycles={idle})")
            pv, pr = prev[v]
            if pv and not pr and not cv:
                raise VIPError(f"{EDGE_ID}: {v} dropped before {r} (valid_hold_until_ready)")
            prev[v] = (cv, cr)
        if since_reset is not None:
            since_reset += 1
'''
