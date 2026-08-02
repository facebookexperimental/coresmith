# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PDKConfig -- Process Design Kit configuration abstraction.

Centralizes all PDK-specific paths, corners, and parameters that are
currently scattered across eda_activities.py, eda_tools.py, config.yaml,
and various TCL scripts.

Usage::

    pdk = PDKConfig.from_yaml("pdk/configs/sky130.yaml", pdk_root=".pdk")
    lib_path = pdk.liberty_path()                 # default corner
    lib_path = pdk.liberty_path("tt_025C_1v80")   # specific corner
    print(pdk.to_summary())                        # for LLM context
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _cells_from_dict(data: dict[str, Any] | None) -> "CellConfig":
    """Build a CellConfig from a plain dict, tolerating missing keys."""
    d = data or {}
    return CellConfig(
        tapcell=d.get("tapcell", ""),
        clkbuf_list=list(d.get("clkbuf_list", []) or []),
        clkbuf_root=d.get("clkbuf_root", ""),
        fillers=list(d.get("fillers", []) or []),
        dont_use=list(d.get("dont_use", []) or []),
    )


def _pnr_from_dict(data: dict[str, Any] | None) -> "PnrConfig":
    """Build a PnrConfig from a plain dict, tolerating missing keys.

    Track values are coerced to ``str`` so a YAML author may write bare numbers
    (``0.23``) OR quoted tokens (``"1.70"``) -- both round-trip to the exact
    token the reference TCL needs.
    """
    d = data or {}
    tracks = []
    for t in d.get("tracks", []) or []:
        tracks.append({k: str(v) for k, v in t.items()})
    pdn = dict(d.get("pdn", {}) or {})
    defaults = PnrConfig()
    return PnrConfig(
        tapcell_distance=int(d.get("tapcell_distance", defaults.tapcell_distance)),
        tracks=tracks,
        pdn=pdn,
        wire_rc_signal_layer=d.get("wire_rc_signal_layer", ""),
        wire_rc_clock_layer=d.get("wire_rc_clock_layer", ""),
        routing_signal=d.get("routing_signal", ""),
        routing_clock=d.get("routing_clock", ""),
        max_fanout=int(d.get("max_fanout", defaults.max_fanout)),
        global_place_pad=int(d.get("global_place_pad", defaults.global_place_pad)),
    )


@dataclass
class CornerConfig:
    """PVT corner configuration."""

    name: str
    liberty: str  # path template with {pdk_root} placeholder
    temperature: float  # degrees C
    voltage: float  # volts
    process: str = "typical"  # typical / slow / fast

    def resolve_liberty(self, pdk_root: str) -> str:
        """Resolve the liberty file path by substituting {pdk_root}."""
        return self.liberty.replace("{pdk_root}", pdk_root)


@dataclass
class CellConfig:
    """Named standard cells the backend flow references by literal name.

    These are the cell-family-specific names a PnR / synthesis flow must emit
    verbatim (a Synopsys/ASAP7 deployment supplies its own). Empty defaults keep
    a config that predates these keys loading unchanged.
    """

    tapcell: str = ""            # tap cell master (OpenROAD ``tapcell``)
    clkbuf_list: list[str] = field(default_factory=list)   # CTS buffer list
    clkbuf_root: str = ""        # CTS root buffer
    fillers: list[str] = field(default_factory=list)       # filler/decap order
    dont_use: list[str] = field(default_factory=list)      # resizer exclusions


@dataclass
class PnrConfig:
    """PnR floorplan / PDN / routing parameters that are PDK-derived data.

    Long-tail numeric tables (tracks, PDN stripe geometry) are kept as
    string-valued tokens so a generated TCL byte-reproduces the reference flow
    regardless of float formatting (``1.70`` must not become ``1.7``). Empty
    defaults keep a pre-existing config loading unchanged.
    """

    tapcell_distance: int = 14
    # Each track: {layer, x_offset, x_pitch, y_offset, y_pitch} -- all STRING
    # tokens so the emitted make_tracks line is byte-identical.
    tracks: list[dict[str, str]] = field(default_factory=list)
    # PDN stripe spec (all string tokens where they land in TCL numerics).
    pdn: dict[str, Any] = field(default_factory=dict)
    wire_rc_signal_layer: str = ""
    wire_rc_clock_layer: str = ""
    routing_signal: str = ""     # e.g. "met1-met4"
    routing_clock: str = ""      # e.g. "met3-met4"
    max_fanout: int = 16
    global_place_pad: int = 2


@dataclass
class PDKConfig:
    """Process Design Kit configuration.

    Holds all PDK-specific paths and parameters needed by the EDA tools
    and architecture agents. Loaded from a YAML config file.
    """

    name: str  # e.g. "sky130"
    process_nm: int  # e.g. 130
    std_cell_library: str  # e.g. "sky130_fd_sc_hd"
    site_name: str  # e.g. "unithd"
    supply_voltage: float  # e.g. 1.8
    default_corner: str  # e.g. "tt_025C_1v80"
    corners: dict[str, CornerConfig] = field(default_factory=dict)
    lef_path: str = ""  # LEF file path template
    tech_lef_path: str = ""  # tech LEF path template
    pdk_root: str = ""  # resolved absolute PDK root path
    # Backend cell + PnR data (optional; empty for configs predating these keys).
    cells: CellConfig = field(default_factory=CellConfig)
    pnr: PnrConfig = field(default_factory=PnrConfig)

    def liberty_path(self, corner: str | None = None) -> str:
        """Resolve liberty file path for a corner.

        Args:
            corner: Corner name (e.g. "tt_025C_1v80"). Defaults to default_corner.

        Returns:
            Absolute path to the liberty file.

        Raises:
            KeyError: If the corner is not defined.
        """
        corner = corner or self.default_corner
        if corner not in self.corners:
            raise KeyError(
                f"Corner '{corner}' not found in PDK '{self.name}'. "
                f"Available: {list(self.corners)}"
            )
        return self.corners[corner].resolve_liberty(self.pdk_root)

    def resolve_lef(self) -> str:
        """Resolve LEF file path."""
        return self.lef_path.replace("{pdk_root}", self.pdk_root)

    def resolve_tech_lef(self) -> str:
        """Resolve tech LEF file path."""
        return self.tech_lef_path.replace("{pdk_root}", self.pdk_root)

    def to_summary(self) -> str:
        """Human-readable summary suitable for LLM context."""
        corner_names = list(self.corners.keys())
        return (
            f"{self.name} ({self.process_nm}nm), "
            f"{self.supply_voltage}V supply, "
            f"library: {self.std_cell_library}, "
            f"site: {self.site_name}, "
            f"corners: {corner_names}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return {
            "name": self.name,
            "process_nm": self.process_nm,
            "std_cell_library": self.std_cell_library,
            "site_name": self.site_name,
            "supply_voltage": self.supply_voltage,
            "default_corner": self.default_corner,
            "corners": {
                name: {
                    "liberty": c.liberty,
                    "temperature": c.temperature,
                    "voltage": c.voltage,
                    "process": c.process,
                }
                for name, c in self.corners.items()
            },
            "lef_path": self.lef_path,
            "tech_lef_path": self.tech_lef_path,
            "pdk_root": self.pdk_root,
            "cells": asdict(self.cells),
            "pnr": asdict(self.pnr),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PDKConfig:
        """Deserialize from a plain dict (backward compatible: cells/pnr optional)."""
        corners = {}
        for name, cdata in data.get("corners", {}).items():
            corners[name] = CornerConfig(
                name=name,
                liberty=cdata["liberty"],
                temperature=cdata["temperature"],
                voltage=cdata["voltage"],
                process=cdata.get("process", "typical"),
            )
        return cls(
            name=data["name"],
            process_nm=data["process_nm"],
            std_cell_library=data["std_cell_library"],
            site_name=data["site_name"],
            supply_voltage=data["supply_voltage"],
            default_corner=data["default_corner"],
            corners=corners,
            lef_path=data.get("lef_path", ""),
            tech_lef_path=data.get("tech_lef_path", ""),
            pdk_root=data.get("pdk_root", ""),
            cells=_cells_from_dict(data.get("cells")),
            pnr=_pnr_from_dict(data.get("pnr")),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str, pdk_root: str = "") -> PDKConfig:
        """Load PDK config from a YAML file.

        Args:
            yaml_path: Path to the YAML config file.
            pdk_root: Root directory of the PDK installation.
                      If empty, uses PDK_ROOT env var or defaults to ".pdk".

        Returns:
            Configured PDKConfig instance with resolved paths.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not pdk_root:
            pdk_root = os.environ.get("PDK_ROOT", ".pdk")
        pdk_root = str(Path(pdk_root).resolve())

        corners: dict[str, CornerConfig] = {}
        for name, cdata in data.get("corners", {}).items():
            corners[name] = CornerConfig(
                name=name,
                liberty=cdata["liberty"],
                temperature=cdata.get("temperature", 25),
                voltage=cdata.get("voltage", data.get("supply_voltage", 1.8)),
                process=cdata.get("process", "typical"),
            )

        return cls(
            name=data["name"],
            process_nm=data["process_nm"],
            std_cell_library=data["std_cell_library"],
            site_name=data["site_name"],
            supply_voltage=data["supply_voltage"],
            default_corner=data["default_corner"],
            corners=corners,
            lef_path=data.get("lef", ""),
            tech_lef_path=data.get("tech_lef", ""),
            pdk_root=pdk_root,
            cells=_cells_from_dict(data.get("cells")),
            pnr=_pnr_from_dict(data.get("pnr")),
        )
