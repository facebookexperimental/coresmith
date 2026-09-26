# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical AMBA signal lists -- the single source of truth for the fabric
generator's ports, the ``axi4`` / ``axi_lite`` / ``apb`` contract families,
the conformance gate and the VIP registry.

Directions are from the MASTER's point of view (``m2s`` = master drives).
Widths are expressions over ``AW`` (address), ``DW`` (data), ``IW`` (id),
``UW`` (user).
"""
from __future__ import annotations

AXI4 = [
    # AW
    ("awvalid", "m2s", "1"), ("awready", "s2m", "1"), ("awaddr", "m2s", "AW"), ("awid", "m2s", "IW"),
    ("awlen", "m2s", "8"), ("awsize", "m2s", "3"), ("awburst", "m2s", "2"), ("awlock", "m2s", "1"),
    ("awcache", "m2s", "4"), ("awprot", "m2s", "3"), ("awqos", "m2s", "4"), ("awregion", "m2s", "4"),
    ("awuser", "m2s", "UW"),
    # W
    ("wvalid", "m2s", "1"), ("wready", "s2m", "1"), ("wdata", "m2s", "DW"), ("wstrb", "m2s", "DW/8"),
    ("wlast", "m2s", "1"), ("wuser", "m2s", "UW"),
    # B
    ("bvalid", "s2m", "1"), ("bready", "m2s", "1"), ("bid", "s2m", "IW"), ("bresp", "s2m", "2"),
    ("buser", "s2m", "UW"),
    # AR
    ("arvalid", "m2s", "1"), ("arready", "s2m", "1"), ("araddr", "m2s", "AW"), ("arid", "m2s", "IW"),
    ("arlen", "m2s", "8"), ("arsize", "m2s", "3"), ("arburst", "m2s", "2"), ("arlock", "m2s", "1"),
    ("arcache", "m2s", "4"), ("arprot", "m2s", "3"), ("arqos", "m2s", "4"), ("arregion", "m2s", "4"),
    ("aruser", "m2s", "UW"),
    # R
    ("rvalid", "s2m", "1"), ("rready", "m2s", "1"), ("rdata", "s2m", "DW"), ("rid", "s2m", "IW"),
    ("rresp", "s2m", "2"), ("rlast", "s2m", "1"), ("ruser", "s2m", "UW"),
]

AXI_LITE = [
    ("awvalid", "m2s", "1"), ("awready", "s2m", "1"), ("awaddr", "m2s", "AW"), ("awprot", "m2s", "3"),
    ("wvalid", "m2s", "1"), ("wready", "s2m", "1"), ("wdata", "m2s", "DW"), ("wstrb", "m2s", "DW/8"),
    ("bvalid", "s2m", "1"), ("bready", "m2s", "1"), ("bresp", "s2m", "2"),
    ("arvalid", "m2s", "1"), ("arready", "s2m", "1"), ("araddr", "m2s", "AW"), ("arprot", "m2s", "3"),
    ("rvalid", "s2m", "1"), ("rready", "m2s", "1"), ("rdata", "s2m", "DW"), ("rresp", "s2m", "2"),
]

APB = [
    ("psel", "m2s", "1"), ("penable", "m2s", "1"), ("pwrite", "m2s", "1"), ("paddr", "m2s", "AW"),
    ("pprot", "m2s", "3"), ("pwdata", "m2s", "DW"), ("pstrb", "m2s", "DW/8"),
    ("pready", "s2m", "1"), ("prdata", "s2m", "DW"), ("pslverr", "s2m", "1"),
]

CHANNELS: dict[str, list[tuple[str, str, str]]] = {"axi4": AXI4, "axi_lite": AXI_LITE, "apb": APB}

# Handshake pairs per family (valid, ready) for the contract/VIP layer.
HANDSHAKES: dict[str, list[tuple[str, str]]] = {
    "axi4": [("awvalid", "awready"), ("wvalid", "wready"), ("bvalid", "bready"),
             ("arvalid", "arready"), ("rvalid", "rready")],
    "axi_lite": [("awvalid", "awready"), ("wvalid", "wready"), ("bvalid", "bready"),
                 ("arvalid", "arready"), ("rvalid", "rready")],
    "apb": [("psel", "pready")],
}


def width_bits(expr: str, *, AW: int, DW: int, IW: int, UW: int) -> int:
    return int(eval(expr, {"__builtins__": {}}, {"AW": AW, "DW": DW, "IW": IW, "UW": UW}))  # noqa: S307 - fixed vocabulary


def port_name(channel: str, signal: str) -> str:
    """``<channel>_<signal>`` -- the same rule as contract_conformance.canonical_port."""
    return f"{channel}_{signal}" if channel else signal


def ports_for(family: str, channel: str, *, role: str, AW: int, DW: int, IW: int, UW: int) -> list[dict]:
    """Flat port list for one side of a bus. ``role`` = 'master' (block
    drives m2s signals) or 'slave'. Returns [{name, dir(input|output), width}]."""
    out = []
    for sig, d, w in CHANNELS[family]:
        drives = (d == "m2s") if role == "master" else (d == "s2m")
        out.append({"name": port_name(channel, sig), "signal": sig,
                    "dir": "output" if drives else "input",
                    "width": width_bits(w, AW=AW, DW=DW, IW=IW, UW=UW)})
    return out
