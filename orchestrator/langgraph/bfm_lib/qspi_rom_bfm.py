# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Passive QSPI/SPI ROM-responder BFM (the DUT-mastered second bus).

An accelerator that reads its operands from an external ROM/flash is the MASTER
of a second bus: it drives ``rom_csn`` / ``rom_sck`` / ``rom_io*`` and expects a
device to answer with data. The QSPI-*slave* master BFM models only the host
side; without a device on the ROM bus the DUT stalls waiting for operands, so a
deterministic DV that omits this responder cannot exercise the real datapath.

This BFM is the missing device: a PASSIVE responder that WATCHES the DUT-driven
``csn``/``sck``/``io`` pin group, decodes a standard SPI-flash READ command
(``cmd`` byte, then ``addr_bytes`` address bytes, then ``dummy_cycles``), and
shifts back bytes from a supplied operand buffer -- MSB-first, sampled on the
DUT's SCK edges, exactly mirroring the master BFM's framing so the two are a
matched pair. It never drives the command/address lanes and never inspects DUT
internals: it is a pure function of the ROM contract + the operand buffer.

Behaviour is deterministic: the same contract + same buffer yield byte-identical
responses, so a run is reproducible and a non-conformant DUT (wrong cmd, wrong
address framing, missing dummy cycles) de-aligns and is caught.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class QSPIRomContract:
    """Framing + pin map for a DUT-mastered SPI/QSPI ROM read bus (DUT-blind).

    Pin BITS are indices into the ROM pin group's vectors when the ROM bus is a
    packed bus; when the top exposes discrete ``rom_csn``/``rom_sck``/``rom_io*``
    scalars the responder binds those names directly (see ``pin_prefix``).
    """
    pin_prefix: str = "rom"       # discrete pins rom_csn / rom_sck / rom_io0..3
    read_cmd: int = 0x03          # standard SPI-flash READ (0x0B fast-read also common)
    addr_bytes: int = 3           # 24-bit address is the flash default
    dummy_cycles: int = 0         # fast-read inserts dummy cycles after address
    lanes: int = 1                # 1 = single (io0 out), 4 = quad (io0..3)
    sample_on_rising: bool = True  # DUT shifts on one edge; sample on the other
    clk_name: str = "wb_clk_i"
    timeout_cyc: int = 1_000_000

    def fingerprint(self) -> str:
        import hashlib
        import json as _json
        blob = _json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


class QSPIRomResponderBFM:
    """Passive ROM device on the DUT-mastered bus. Answers READ from a buffer.

    Bind by discrete pin names ``<prefix>_csn`` / ``<prefix>_sck`` /
    ``<prefix>_io0..3`` and (for single-lane) ``<prefix>_io1`` as the device
    output (MISO). Drive ONLY the output lane(s); the csn/sck/cmd/addr lanes are
    DUT-driven and only sampled. ``rom_data`` is the operand buffer the DUT reads.
    """

    def __init__(self, dut, contract: QSPIRomContract, rom_data: bytes):
        self.dut = dut
        self.c = contract
        self.rom = bytes(rom_data)
        p = contract.pin_prefix
        self._csn = getattr(dut, f"{p}_csn", None)
        self._sck = getattr(dut, f"{p}_sck", None)
        self._io0 = getattr(dut, f"{p}_io0", None)  # DUT->dev (MOSI / cmd/addr)
        self._io1 = getattr(dut, f"{p}_io1", None)  # dev->DUT (MISO) in single-lane
        self._io = [getattr(dut, f"{p}_io{i}", None) for i in range(4)]
        self.reads: list[tuple[int, int]] = []      # (addr, nbytes) served

    def _miso(self):
        # single-lane: device drives io1 (MISO); quad: io0..3
        return self._io1 if self.c.lanes == 1 else None

    async def run(self):
        """Serve READ transactions until the sim ends. Never returns normally."""

        clk = getattr(self.dut, self.c.clk_name)
        miso = self._miso()
        # idle the output lane so the DUT sees a defined level before a read.
        if miso is not None:
            miso.value = 0
        while True:
            # wait for csn to fall (transaction start)
            await self._wait_csn_low(clk)
            try:
                await self._serve_one(clk)
            except _Deassert:
                continue

    async def _wait_csn_low(self, clk):
        from cocotb.triggers import RisingEdge
        for _ in range(self.c.timeout_cyc):
            await RisingEdge(clk)
            if self._csn is not None and int(self._csn.value) == 0:
                return

    def _csn_high(self) -> bool:
        return self._csn is not None and int(self._csn.value) == 1

    async def _sck_active_edge(self, clk):
        """Advance to the DUT's active SCK edge (rising by convention)."""
        from cocotb.triggers import RisingEdge
        prev = int(self._sck.value) if self._sck is not None else 0
        for _ in range(self.c.timeout_cyc):
            await RisingEdge(clk)
            if self._csn_high():
                raise _Deassert()
            cur = int(self._sck.value) if self._sck is not None else 0
            rising = prev == 0 and cur == 1
            falling = prev == 1 and cur == 0
            prev = cur
            if (rising if self.c.sample_on_rising else falling):
                return

    async def _shift_in_bits(self, clk, nbits) -> int:
        """Sample nbits from the DUT-driven io0 (single) on each active edge."""
        val = 0
        for _ in range(nbits):
            await self._sck_active_edge(clk)
            bit = int(self._io0.value) & 1 if self._io0 is not None else 0
            val = (val << 1) | bit
        return val

    async def _shift_out_bits(self, clk, byte):
        """Drive nbits of `byte` MSB-first on MISO on each active edge."""
        miso = self._miso()
        for i in range(8):
            await self._sck_active_edge(clk)
            if miso is not None:
                miso.value = (byte >> (7 - i)) & 1

    async def _serve_one(self, clk):
        # cmd byte
        cmd = await self._shift_in_bits(clk, 8)
        if cmd != self.c.read_cmd:
            # not a read we model -> ignore until csn deasserts
            return
        addr = 0
        for _ in range(self.c.addr_bytes):
            addr = (addr << 8) | await self._shift_in_bits(clk, 8)
        for _ in range(self.c.dummy_cycles):
            await self._sck_active_edge(clk)
        # stream data bytes from the buffer until the DUT deasserts csn
        served = 0
        a = addr
        try:
            while True:
                byte = self.rom[a] if 0 <= a < len(self.rom) else 0
                await self._shift_out_bits(clk, byte)
                a += 1
                served += 1
        except _Deassert:
            self.reads.append((addr, served))
            raise


class _Deassert(Exception):
    """Internal: csn deasserted mid-transaction (end of this read)."""
