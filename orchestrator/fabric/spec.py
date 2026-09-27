# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The declarative description of one fabric instance."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
MASTER_PROTOCOLS = ("axi4",)
SLAVE_PROTOCOLS = ("axi4", "axi_lite", "apb")
# axi_xbar LatencyMode (axi_pkg::xbar_latency_e). cut_all_ports registers every
# channel (AW/W/B/AR/R) at both crossbar sides; cut_all_ax leaves W/B/R
# combinational end to end and misses timing on wide fabrics.
LATENCY_MODES = {"cut_all_ports": "CUT_ALL_PORTS", "cut_all_ax": "CUT_ALL_AX"}


def _int(v, default=None):
    if v is None:
        return default
    if isinstance(v, str) and v.lower().startswith("0x"):
        return int(v, 16)
    return int(v)


@dataclass
class FabricMaster:
    name: str
    protocol: str = "axi4"
    id_width: int = 4
    max_outstanding: int = 4      # transactions per direction the fabric tracks


@dataclass
class FabricSlave:
    name: str
    protocol: str = "axi4"
    base: int = 0
    size: int = 0x1000            # bytes; power of two
    data_width: int | None = None  # defaults to the fabric's


@dataclass
class FabricSpec:
    name: str
    masters: list[FabricMaster] = field(default_factory=list)
    slaves: list[FabricSlave] = field(default_factory=list)
    data_width: int = 32
    addr_width: int = 32
    user_width: int = 1
    ordering: str = "per_id"      # per_id | none (an axi_cut per axi4 slave port)
    err_slave: bool = True        # unmapped addresses answer DECERR
    max_outstanding: int = 4      # per slave port
    latency_mode: str = "cut_all_ports"   # see LATENCY_MODES
    # slave_cut: an axi_cut on every slave port. For AXI-Lite/APB ports it registers
    # both sides of the converter: a full-AXI axi_cut before axi_to_axi_lite, an
    # AXI-Lite axi_cut after it, and axi_lite_to_apb's PipelineRequest/Response.
    slave_cut: bool = True

    # ------------------------------------------------------------ helpers
    @property
    def module_name(self) -> str:
        return f"cs_fabric_{self.name}"

    @property
    def mst_id_width(self) -> int:
        """Slave-port ID width: master id width + log2(masters)."""
        n = max(1, len(self.masters))
        return max(m.id_width for m in self.masters) + max(0, (n - 1).bit_length())

    def slave_index(self, name: str) -> int:
        for i, s in enumerate(self.slaves):
            if s.name == name:
                return i
        raise KeyError(name)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not _NAME.match(self.name or ""):
            errs.append(f"fabric name {self.name!r} must be snake_case")
        if not self.masters:
            errs.append("a fabric needs at least one master")
        if not self.slaves:
            errs.append("a fabric needs at least one slave")
        if self.data_width not in (32, 64, 128):
            errs.append(f"data_width {self.data_width} not in (32, 64, 128)")
        if self.addr_width not in (32, 40, 48, 56, 64):
            errs.append(f"addr_width {self.addr_width} unusual")
        if self.ordering not in ("per_id", "none"):
            errs.append(f"ordering {self.ordering!r}")
        if self.latency_mode not in LATENCY_MODES:
            errs.append(f"latency_mode {self.latency_mode!r} not in {tuple(LATENCY_MODES)}")
        names: set[str] = set()
        for m in self.masters:
            if not _NAME.match(m.name or ""):
                errs.append(f"master name {m.name!r} must be snake_case")
            if m.name in names:
                errs.append(f"duplicate port name {m.name!r}")
            names.add(m.name)
            if m.protocol not in MASTER_PROTOCOLS:
                errs.append(f"master {m.name}: protocol {m.protocol!r} (masters are axi4)")
            if not 1 <= int(m.id_width) <= 16:
                errs.append(f"master {m.name}: id_width {m.id_width}")
        if len({m.id_width for m in self.masters}) > 1:
            errs.append("all masters must share one id_width (use axi_id_remap upstream)")
        spans = []
        for s in self.slaves:
            if not _NAME.match(s.name or ""):
                errs.append(f"slave name {s.name!r} must be snake_case")
            if s.name in names:
                errs.append(f"duplicate port name {s.name!r}")
            names.add(s.name)
            if s.protocol not in SLAVE_PROTOCOLS:
                errs.append(f"slave {s.name}: protocol {s.protocol!r}")
            if s.size <= 0 or (s.size & (s.size - 1)):
                errs.append(f"slave {s.name}: size {s.size:#x} must be a power of two")
            if s.base % max(1, s.size):
                errs.append(f"slave {s.name}: base {s.base:#x} not aligned to its size")
            if s.data_width not in (None, self.data_width):
                errs.append(f"slave {s.name}: data-width conversion is not generated yet")
            spans.append((s.base, s.base + s.size, s.name))
        spans.sort()
        for (a0, a1, n0), (b0, b1, n1) in zip(spans, spans[1:]):
            if b0 < a1:
                errs.append(f"slaves {n0} and {n1} overlap ({a0:#x}-{a1:#x} vs {b0:#x}-{b1:#x})")
        return errs

    # ------------------------------------------------------------ (de)serialisation
    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "FabricSpec":
        d = dict(d or {})
        masters = [FabricMaster(name=str(m.get("name")), protocol=str(m.get("protocol") or "axi4"),
                                id_width=_int(m.get("id_width"), 4),
                                max_outstanding=_int(m.get("max_outstanding"), 4))
                   for m in (d.get("masters") or [])]
        slaves = [FabricSlave(name=str(s.get("name")), protocol=str(s.get("protocol") or "axi4"),
                              base=_int(s.get("base"), 0), size=_int(s.get("size"), 0x1000),
                              data_width=_int(s.get("data_width"), None))
                  for s in (d.get("slaves") or [])]
        return cls(name=str(d.get("name") or "soc"), masters=masters, slaves=slaves,
                   data_width=_int(d.get("data_width"), 32), addr_width=_int(d.get("addr_width"), 32),
                   user_width=_int(d.get("user_width"), 1), ordering=str(d.get("ordering") or "per_id"),
                   err_slave=bool(d.get("err_slave", True)),
                   max_outstanding=_int(d.get("max_outstanding"), 4),
                   latency_mode=str(d.get("latency_mode") or "cut_all_ports").lower(),
                   slave_cut=bool(d.get("slave_cut", True)))

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]
