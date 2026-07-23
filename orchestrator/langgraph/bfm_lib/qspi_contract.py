# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""QSPI-slave bus contract parameters.

A :class:`QSPIContract` is the *complete, DUT-blind* description of the external
QSPI-slave bus a chip-top exposes on the Caravel GPIO pins. It is derived from
the architecture artifacts (register map + pin mapping + command set) -- **never**
from the DUT RTL -- and is the sole input to the deterministic master BFM
generator. Same contract in => byte-identical BFM out.

The default values encode the QSPI-slave accelerator chassis contract
(``chassis/accel/PROTOCOL.md`` + ``qspi_host.py``): quad IO, mode 0 (sample on
rising SCK), command 0x02 write / 0x03 read / 0x05 read-status, 24-bit address,
1 dummy byte of bus turnaround before read data, MSB-nibble-first within a byte,
little-endian byte order within a burst.

This is deliberately a *frozen* dataclass: the whole point of the deterministic
BFM is that the pin driver is a pure function of the contract, so the contract
must be immutable + hashable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class QSPIContract:
    """Immutable QSPI-slave bus contract (the master BFM's only input)."""

    # ---- data path --------------------------------------------------------
    data_width: int = 8               # bits per logical data unit (one "byte")
    lanes: int = 4                    # quad IO
    nibble_order: str = "msb_first"   # nibble order within a byte on the lanes
    endianness: str = "little"        # byte order within an IN/OUT burst

    # ---- command set ------------------------------------------------------
    cmd_write: int = 0x02
    cmd_read: int = 0x03
    cmd_status: int = 0x05
    addr_bytes: int = 3               # 24-bit QSPI address space
    dummy_bytes: int = 1             # bus-turnaround dummy bytes before read data
    sample_edge: str = "rising"      # mode-0: both sides sample on rising SCK

    # ---- register / buffer map (byte addresses in the 24-bit space) -------
    ctrl_addr: int = 0x000000        # W: bit0 START, bit1 SOFT_RESET
    status_addr: int = 0x000004      # R: bit0 BUSY, bit1 DONE, bit2 ERROR
    cfg0_addr: int = 0x000008        # RW scalar (e.g. n_blocks / length / mode)
    cfg1_addr: int = 0x00000C        # RW scalar
    # Width (bytes) of the conformance CFG1 write/read-back probe. Default 1:
    # a byte-wide access is universally legal on the byte-addressed map,
    # whereas a 4-byte access spans undecoded bytes on designs whose CFG1 is
    # a narrow register (false-fail). Set 4 only when CFG1 is a full 32-bit
    # scalar and the wider probe is wanted.
    cfg1_width_bytes: int = 1
    in_addr: int = 0x001000          # W input window (auto-inc)
    out_addr: int = 0x002000         # R output window (auto-inc)

    # ---- status field bit positions --------------------------------------
    status_busy_bit: int = 0
    status_done_bit: int = 1
    status_error_bit: int = 2

    # ---- control field bit positions -------------------------------------
    ctrl_start_bit: int = 0
    ctrl_soft_reset_bit: int = 1

    # ---- pin map on the Caravel io_* GPIO --------------------------------
    csn_bit: int = 0                 # io[0]  qspi_csn (host->DUT, active low)
    sck_bit: int = 1                 # io[1]  qspi_sck (host-generated)
    io0_bit: int = 2                 # io[5:2] qspi_io[3:0] (bidir quad lanes)
    irq_bit: int = 6                 # io[6]  irq (DUT->host, optional)

    # ---- clock / reset / timing ------------------------------------------
    clk_name: str = "wb_clk_i"
    rst_name: str = "wb_rst_i"
    reset_active_high: bool = True
    clk_period_ns: int = 20          # 50 MHz core clock
    sck_half_period: int = 4         # SCK half-period in wb_clk cycles
    io_width: int = 38               # Caravel GPIO width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QSPIContract":
        fields = {f for f in cls.__dataclass_fields__}  # noqa: E1101
        return cls(**{k: v for k, v in d.items() if k in fields})

    # A stable, human-auditable fingerprint of the contract. Two runs that
    # classify to the same contract emit byte-identical BFMs; this string is
    # embedded in the generated file header so the determinism is inspectable.
    def fingerprint(self) -> str:
        import hashlib
        import json

        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
