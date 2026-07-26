# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic QSPI-slave *master* BFM (the host-MCU side of the bus).

This is the contract-faithful replacement for the LLM-authored integration/
composition pin driver. It models the external host MCU that drives a
QSPI-slave accelerator over the Caravel GPIO pins, using a FIXED clock schedule
derived only from the :class:`QSPIContract` -- it NEVER sees the DUT RTL, so it
cannot co-tune its timing to a non-conformant DUT's quirks.

Behaviour is a parameterized, byte-for-byte port of the accelerator-chassis reference host
BFM (``chassis/accel/qspi_host.py``): quad IO, mode 0 (drive while SCK low,
sample on rising SCK), MSB-nibble-first within a byte, ``cmd 0x02`` write /
``0x03`` read / ``0x05`` read-status, N dummy bytes of bus turnaround before
read data. Because the schedule is fixed (``sck_half_period`` wb-clks per
half-phase, ``dummy_bytes`` turnaround, sample strictly on the rising edge), a
DUT that stalls between read bytes or drops the dummy-byte turnaround
de-aligns and is caught -- which is exactly the property the co-tuned BFM
lacked (the AES read-serializer bug that passed CoreSmith DV but failed the
fixed ``qspi_host``).

The class is import-clean: it only touches ``cocotb`` *inside* its coroutines,
so it can be imported (and its determinism unit-tested) without cocotb present.
"""

from __future__ import annotations

from .qspi_contract import QSPIContract


class QSPIMasterBFM:
    """Host-MCU QSPI master. Drives ``dut.io_in``; samples ``dut.io_out``.

    Constructed with the DUT handle and a :class:`QSPIContract`. The DUT handle
    is used ONLY for the standard ``io_in`` / ``io_out`` pin bundle and the
    named clock -- the driver logic depends on nothing else about the DUT, by
    construction.
    """

    def __init__(self, dut, contract: QSPIContract):
        self.dut = dut
        self.c = contract
        self.h = contract.sck_half_period
        self._clk = getattr(dut, contract.clk_name)
        # idle: csn high, sck low, io lanes 0
        self._in = 1 << contract.csn_bit
        self.dut.io_in.value = self._in

    # -- low-level GPIO ----------------------------------------------------
    def _drive(self):
        self.dut.io_in.value = self._in

    def _set(self, bit, val):
        if val:
            self._in |= 1 << bit
        else:
            self._in &= ~(1 << bit)

    def _set_io(self, nib):
        self._in = (self._in & ~(0xF << self.c.io0_bit)) | (
            (nib & 0xF) << self.c.io0_bit
        )

    def _read_io(self) -> int:
        return (int(self.dut.io_out.value) >> self.c.io0_bit) & 0xF

    async def _tick(self, n=None):
        from cocotb.triggers import ClockCycles

        await ClockCycles(self._clk, n if n is not None else self.h)

    # -- one quad-nibble out (host drives), DUT samples on rising SCK ------
    async def _shift_out(self, nib):
        self._set(self.c.sck_bit, 0)
        self._set_io(nib)
        self._drive()
        await self._tick()
        self._set(self.c.sck_bit, 1)
        self._drive()          # rising edge: DUT samples
        await self._tick()

    # -- one quad-nibble in (DUT drives), host samples on rising SCK -------
    async def _shift_in(self) -> int:
        self._set(self.c.sck_bit, 0)
        self._drive()          # DUT sets read data while SCK low
        await self._tick()
        self._set(self.c.sck_bit, 1)
        self._drive()
        await self._tick()
        return self._read_io()

    async def _byte_out(self, b):
        if self.c.nibble_order == "msb_first":
            await self._shift_out((b >> 4) & 0xF)
            await self._shift_out(b & 0xF)
        else:
            await self._shift_out(b & 0xF)
            await self._shift_out((b >> 4) & 0xF)

    async def _byte_in(self) -> int:
        a = await self._shift_in()
        b = await self._shift_in()
        if self.c.nibble_order == "msb_first":
            return ((a & 0xF) << 4) | (b & 0xF)
        return ((b & 0xF) << 4) | (a & 0xF)

    async def _cmd_addr(self, cmd, addr):
        await self._byte_out(cmd)
        for i in range(self.c.addr_bytes - 1, -1, -1):
            await self._byte_out((addr >> (8 * i)) & 0xFF)

    def _select(self, on):
        self._set(self.c.csn_bit, 0 if on else 1)
        if not on:
            self._set(self.c.sck_bit, 0)
            self._set_io(0)
        self._drive()

    # -- public transaction API -------------------------------------------
    async def write(self, addr: int, data: bytes):
        """cmd 0x02: write ``data`` to ``addr`` (auto-incrementing), little-endian."""
        self._select(True)
        await self._cmd_addr(self.c.cmd_write, addr)
        for b in data:
            await self._byte_out(b)
        self._select(False)
        await self._tick(2)

    async def read(self, addr: int, n: int) -> bytes:
        """cmd 0x03: read ``n`` bytes from ``addr`` after the dummy turnaround."""
        self._select(True)
        await self._cmd_addr(self.c.cmd_read, addr)
        for _ in range(self.c.dummy_bytes):
            await self._byte_out(0x00)     # bus turnaround: DUT takes the bus
        out = bytes([await self._byte_in() for _ in range(n)])
        self._select(False)
        await self._tick(2)
        return out

    async def read_status(self) -> int:
        """cmd 0x05: read the 1-byte STATUS register after the dummy turnaround."""
        self._select(True)
        await self._byte_out(self.c.cmd_status)
        for _ in range(self.c.dummy_bytes):
            await self._byte_out(0x00)
        st = await self._byte_in()
        self._select(False)
        await self._tick(2)
        return st

    async def send_opcode_only(self, opcode: int):
        """Drive a bare command byte (no address / data) then deselect.

        Used by the bus-protocol conformance DV to probe UNKNOWN-opcode handling:
        a conformant frontend must return to idle (and, per the chassis protocol,
        raise ``STATUS.ERROR``) rather than wedge. DUT-blind: same fixed schedule
        as every other transaction.
        """
        self._select(True)
        await self._byte_out(opcode & 0xFF)
        self._select(False)
        await self._tick(2)

    # -- register helpers (contract reg map) ------------------------------
    async def write_reg(self, addr: int, value: int, width_bytes: int = 4):
        """Write a little-endian scalar register."""
        await self.write(addr, bytes((value >> (8 * i)) & 0xFF for i in range(width_bytes)))

    async def start(self):
        """Pulse CTRL.START."""
        await self.write_reg(
            self.c.ctrl_addr, 1 << self.c.ctrl_start_bit, 4
        )

    async def wait_done(self, timeout_wbclks: int = 5_000_000) -> bool:
        """Poll STATUS.DONE via cmd 0x05. Returns True on done, False on
        timeout or a TERMINAL error.

        DONE (bit1) wins whenever set. A TERMINAL error is ERROR (bit2) asserted
        with BUSY (bit0) clear *after the op has actually started* (BUSY was
        observed high at least once) and DONE never arriving. Before BUSY is
        first seen, ERROR is ignored -- some designs hold ERROR undefined out of
        reset until the first completion, so honoring it pre-BUSY would
        spuriously abort. This keeps the poll contract-faithful (a real terminal
        error still fails closed) without co-tuning to any DUT.
        """
        waited = 0
        step = 20
        done_bit = 1 << self.c.status_done_bit
        busy_bit = 1 << self.c.status_busy_bit
        err_bit = 1 << self.c.status_error_bit
        seen_busy = False
        while waited < timeout_wbclks:
            st = await self.read_status()
            if st & done_bit:
                return True
            if st & busy_bit:
                seen_busy = True
            elif seen_busy and (st & err_bit):
                return False
            await self._tick(step)
            waited += step + 32 * self.h
        return False

    async def reset_dut(self, cycles: int = 8):
        """Assert then deassert the DUT reset for ``cycles`` clocks each."""
        from cocotb.triggers import ClockCycles

        rst = getattr(self.dut, self.c.rst_name)
        active = 1 if self.c.reset_active_high else 0
        rst.value = active
        self.dut.io_in.value = self._in
        await ClockCycles(self._clk, cycles)
        rst.value = 1 - active
        await ClockCycles(self._clk, cycles)
