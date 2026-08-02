# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR1: sky130 deployment path/binary parity + the deployment registry.

Byte-identical parity is asserted against the *old* backend_helpers formula,
reproduced here so a drift in the moved code fails loudly. No real PDK required:
paths are pure ``Path`` composition over a synthetic sky130A/sky130B tree.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from orchestrator.pdk.deployments import sky130
from orchestrator.pdk.registry import (
    get_deployment,
    load_deployment,
    reset_deployment_cache,
)

_STD = "sky130_fd_sc_hd"

# The eight PDK path constants, exactly as backend_helpers.py:56-63 built them
# (the parity oracle -- do NOT refactor to call sky130).
def _old_formula(pdk_root: Path, variant: str) -> dict[str, Path]:
    p = pdk_root / variant
    return {
        "tech_lef": p / "libs.ref" / _STD / "techlef" / f"{_STD}__nom.tlef",
        "cell_lef": p / "libs.ref" / _STD / "lef" / f"{_STD}.lef",
        "liberty": p / "libs.ref" / _STD / "lib" / f"{_STD}__tt_025C_1v80.lib",
        "cell_gds": p / "libs.ref" / _STD / "gds" / f"{_STD}.gds",
        "cell_spice": p / "libs.ref" / _STD / "spice" / f"{_STD}.spice",
        "magic_rc": p / "libs.tech" / "magic" / f"{variant}.magicrc",
        "netgen_setup": p / "libs.tech" / "netgen" / "setup.tcl",
        "rcx_rules": p / "libs.tech" / "rcx" / "sky130hd_rcx_patterns.rules",
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_deployment_cache()
    yield
    reset_deployment_cache()


def _make_variant_tree(root: Path, variant: str) -> Path:
    """Create the minimal directory skeleton for one PDK variant."""
    v = root / variant
    for sub in ("libs.ref", "libs.tech"):
        (v / sub).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Path parity
# ---------------------------------------------------------------------------
class TestSky130PathParity:
    def test_sky130A_paths_byte_identical(self, tmp_path):
        root = _make_variant_tree(tmp_path / "pdk", "sky130A")
        got = sky130._resolve_paths(root)
        exp = _old_formula(root, "sky130A")
        assert got.variant == "sky130A"
        for key, path in exp.items():
            assert str(getattr(got, key)) == str(path), key

    def test_sky130B_variant_selected(self, tmp_path):
        root = _make_variant_tree(tmp_path / "pdk", "sky130B")
        got = sky130._resolve_paths(root)
        exp = _old_formula(root, "sky130B")
        assert got.variant == "sky130B"
        assert str(got.magic_rc) == str(exp["magic_rc"])  # variant in filename
        assert str(got.liberty) == str(exp["liberty"])

    def test_variant_A_precedence_when_both_present(self, tmp_path):
        root = tmp_path / "pdk"
        _make_variant_tree(root, "sky130A")
        _make_variant_tree(root, "sky130B")
        assert sky130._pdk_variant(root) == "sky130A"

    def test_variant_default_when_absent(self, tmp_path):
        assert sky130._pdk_variant(tmp_path / "empty") == "sky130A"

    def test_deployment_reads_pdk_root_env(self, tmp_path, monkeypatch):
        # A fresh deployment must resolve against the monkeypatched PDK_ROOT.
        root = _make_variant_tree(tmp_path / "pdk", "sky130A")
        monkeypatch.setenv("PDK_ROOT", str(root))
        reset_deployment_cache()
        dep = sky130.Sky130Deployment()
        exp = _old_formula(root, "sky130A")
        assert str(dep.tech_lef) == str(exp["tech_lef"])
        assert str(dep.liberty) == str(exp["liberty"])


# ---------------------------------------------------------------------------
# Tool-binary resolution parity + overrides
# ---------------------------------------------------------------------------
class TestToolBinaryResolution:
    def test_reexport_identity_backend_helpers(self):
        import orchestrator.langgraph.backend_helpers as bh
        for name in ("TECH_LEF", "CELL_LEF", "LIBERTY", "CELL_GDS", "CELL_SPICE",
                     "MAGIC_RC", "NETGEN_SETUP", "RCX_RULES", "OPENROAD_BIN",
                     "MAGIC_BIN", "NETGEN_BIN", "KLAYOUT_BIN", "_STD_CELL"):
            assert str(getattr(bh, name)) == str(getattr(sky130, name)), name

    def test_default_resolves_to_nix_wrapper(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_BACKEND_OPENROAD", raising=False)
        got = sky130._resolve_tool("openroad_binary", "scripts/openroad-nix.sh")
        assert got.endswith("openroad-nix.sh")
        assert Path(got).exists()

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_BACKEND_OPENROAD", "/custom/openroad")
        assert sky130._resolve_tool("openroad_binary", "scripts/openroad-nix.sh") \
            == "/custom/openroad"

    def test_yosys_env_override(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_BACKEND_YOSYS", "/opt/x/yosys")
        assert sky130._resolve_yosys() == "/opt/x/yosys"

    def test_yosys_path_hit_wins_when_no_env(self, monkeypatch, tmp_path):
        # Hermetic: a fake yosys on a controlled PATH must win over the
        # nix-wrapper fallback (no dependency on the host having yosys).
        monkeypatch.delenv("CORESMITH_BACKEND_YOSYS", raising=False)
        fake = tmp_path / "yosys"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert sky130._resolve_yosys() == str(fake)

    def test_yosys_off_path_falls_back_to_wrapper_or_bare(self, monkeypatch,
                                                          tmp_path):
        # With no env override and no yosys on PATH (empty dir), resolution
        # must land on the configured/nix wrapper script or the bare name --
        # never raise. CI runners have no yosys, so this is the branch they take.
        monkeypatch.delenv("CORESMITH_BACKEND_YOSYS", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        resolved = sky130._resolve_yosys()
        assert resolved == "yosys" or resolved.endswith(("yosys-nix.sh", "yosys"))


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------
_ONE_FILE_DEPLOYMENT = textwrap.dedent(
    """
    from orchestrator.pdk.base import Deployment
    from orchestrator.pdk.pdk_config import PDKConfig

    class _Custom(Deployment):
        name = "custom_test_dep"
        @property
        def pdk(self):
            return PDKConfig(name="c", process_nm=7, std_cell_library="x",
                             site_name="s", supply_voltage=0.8, default_corner="tt")
        def tools(self):
            return {}

    DEPLOYMENT = _Custom()
    """
)


class TestRegistry:
    def test_default_is_sky130(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DEPLOYMENT", raising=False)
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "load_config", lambda: {})
        reset_deployment_cache()
        assert get_deployment().name == "sky130"

    def test_env_builtin_name(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "mock")
        reset_deployment_cache()
        assert get_deployment().name == "mock"

    def test_config_key(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DEPLOYMENT", raising=False)
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "load_config", lambda: {"deployment": "mock"})
        reset_deployment_cache()
        assert get_deployment().name == "mock"

    def test_env_wins_over_config(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "sky130")
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "load_config", lambda: {"deployment": "mock"})
        reset_deployment_cache()
        assert get_deployment().name == "sky130"

    def test_env_filesystem_path(self, tmp_path, monkeypatch):
        dep_file = tmp_path / "my_dep.py"
        dep_file.write_text(_ONE_FILE_DEPLOYMENT)
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", str(dep_file))
        reset_deployment_cache()
        dep = get_deployment()
        assert dep.name == "custom_test_dep"
        assert dep.pdk.process_nm == 7

    def test_bad_path_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "/nonexistent/does_not_exist.py")
        reset_deployment_cache()
        with pytest.raises(ValueError, match="not found"):
            get_deployment()

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="unknown deployment"):
            load_deployment("not_a_real_deployment")

    def test_missing_deployment_attr_raises(self, tmp_path):
        f = tmp_path / "empty_dep.py"
        f.write_text("X = 1\n")
        with pytest.raises(ValueError, match="DEPLOYMENT"):
            load_deployment(str(f))

    def test_cache_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "mock")
        reset_deployment_cache()
        assert get_deployment() is get_deployment()
