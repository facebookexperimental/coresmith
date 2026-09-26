# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The contract facts a VIP is rendered from, in the conformance gate's
port spelling (``contract_conformance.signal_specs`` / ``canonical_port``),
so the VIP drives exactly the ports the gate demands of the RTL."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from orchestrator.architecture.specialists.contract_timing import default_timing, normalize_timing
from orchestrator.langgraph.contract_conformance import canonical_port, channel_base, signal_specs

FAMILIES = ("axi_stream", "srdy_drdy", "req_resp", "mem_write", "valid_only", "static")
_HANDSHAKE = {"axi_stream": ("tvalid", "tready"), "srdy_drdy": ("srdy", "drdy"),
              "valid_only": ("valid", None), "req_resp": ("req_valid", "req_gnt"),
              "mem_write": ("we", None), "static": (None, None)}
_RESPONSE = {"req_resp": ("rsp_valid",)}


def _ident(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s or "")


@dataclass
class Signal:
    name: str          # contract signal name
    width: int | None
    kind: str          # field | sideband | handshake
    dir: str = ""      # producer->consumer | consumer->producer | ""

    @property
    def to_producer(self) -> bool:
        return "consumer->producer" in (self.dir or "") or self.name in ("tready", "drdy", "ready",
                                                                            "req_gnt", "gnt")


@dataclass
class SideSpec:
    """The VIP as seen from ONE block: its channel prefix and port names."""
    block: str
    role: str                       # producer | consumer
    channel: str                    # channel base (port prefix)
    ports: dict[str, str] = field(default_factory=dict)   # signal -> RTL port name

    def port(self, signal: str) -> str:
        return self.ports.get(signal, f"{self.channel}_{signal}" if self.channel else signal)


@dataclass
class EdgeContract:
    edge_id: str
    family: str
    producer: str
    consumer: str
    producer_port: str
    consumer_port: str
    data_width: int
    fields: list[dict]
    signals: list[Signal]
    timing: dict
    enums: dict
    raw: dict

    @classmethod
    def from_contract(cls, edge: dict) -> "EdgeContract":
        fam = str(edge.get("handshake_protocol") or "").strip().lower() or "valid_only"
        e = dict(edge)
        e["timing"] = dict(e["timing"]) if isinstance(e.get("timing"), dict) else default_timing(fam, e)
        normalize_timing(e)  # always: fills family defaults the specialist left out
        sigs: list[Signal] = []
        for s in signal_specs(e):
            try:
                w = int(s["width"]) if s.get("width") not in ("", None) else None
            except ValueError:
                w = None
            sigs.append(Signal(name=str(s["name"]), width=w, kind=str(s.get("kind") or "field"),
                               dir=str(s.get("dir") or "")))
        names = {s.name for s in sigs}
        for extra in _RESPONSE.get(fam, ()):
            if extra not in names and not any(n in names for n in ("rvalid", "rsp_valid", "resp_valid")):
                sigs.append(Signal(name=extra, width=1, kind="handshake", dir="consumer->producer"))
        try:
            dw = int(e.get("data_width_bits") or 0)
        except (TypeError, ValueError):
            dw = 0
        enums = {}
        for en in ((e.get("representations") or {}).get("enums") or []):
            if isinstance(en, dict) and en.get("name"):
                enums[str(en["name"])] = dict(en.get("values") or {})
        return cls(edge_id=str(e.get("edge_id") or f"{e.get('producer_block')}__to__{e.get('consumer_block')}"),
                   family=fam, producer=str(e.get("producer_block") or ""),
                   consumer=str(e.get("consumer_block") or ""),
                   producer_port=str(e.get("producer_port") or ""),
                   consumer_port=str(e.get("consumer_port") or ""),
                   data_width=dw, fields=list(e.get("fields") or []), signals=sigs,
                   timing=dict(e["timing"]), enums=enums, raw=e)

    # ---- naming
    @property
    def module_name(self) -> str:
        return _ident(self.edge_id)

    def side(self, role: str) -> SideSpec:
        block = self.producer if role == "producer" else self.consumer
        chan_raw = self.producer_port if role == "producer" else self.consumer_port
        chan = channel_base(chan_raw) or ""
        ports = {}
        for s in self.signals:
            port, _bare = canonical_port(chan, s.name)
            ports[s.name] = port or s.name
        return SideSpec(block=block, role=role, channel=chan, ports=ports)

    # ---- handshake facts
    @property
    def valid_signal(self) -> str | None:
        v = _HANDSHAKE.get(self.family, (None, None))[0]
        if self.family == "req_resp":
            for cand in ("req_valid", "valid", "req"):
                if any(s.name == cand for s in self.signals):
                    return cand
            return "req_valid"
        if self.family == "mem_write":
            for cand in ("we", "write_enable", "wen", "write_commit", "wr_en"):
                if any(s.name == cand for s in self.signals):
                    return cand
            return None
        return v

    @property
    def ready_signal(self) -> str | None:
        r = _HANDSHAKE.get(self.family, (None, None))[1]
        if self.family == "req_resp":
            for cand in ("req_gnt", "gnt", "req_ready", "ready"):
                if any(s.name == cand for s in self.signals):
                    return cand
            return None  # request accepted on valid alone (no ready)
        return r

    @property
    def response_valid_signal(self) -> str | None:
        if self.family != "req_resp":
            return None
        for cand in ("rsp_valid", "rvalid", "resp_valid"):
            if any(s.name == cand for s in self.signals):
                return cand
        return "rsp_valid"

    @property
    def last_signal(self) -> str | None:
        b = (self.timing or {}).get("burst") or {}
        return b.get("last_signal")

    @property
    def payload_signals(self) -> list[Signal]:
        skip = {self.valid_signal, self.ready_signal, self.response_valid_signal, self.last_signal}
        return [s for s in self.signals if s.kind in ("field", "sideband") and s.name not in skip
                and not s.to_producer]

    @property
    def response_signals(self) -> list[Signal]:
        if self.family != "req_resp":
            return []
        out = []
        for s in self.signals:
            n = s.name.lower()
            if s.name == self.response_valid_signal or s.name == self.ready_signal:
                continue
            if s.to_producer or n.startswith(("rdata", "rsp", "resp", "rresp")) or n in ("fault", "error"):
                out.append(s)
        return out

    def fingerprint(self) -> str:
        key = {"edge_id": self.edge_id, "family": self.family, "timing": self.timing,
               "signals": [(s.name, s.width, s.kind, s.dir) for s in self.signals],
               "producer_port": self.producer_port, "consumer_port": self.consumer_port}
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
