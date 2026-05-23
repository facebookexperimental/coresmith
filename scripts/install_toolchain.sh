#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# install_toolchain.sh -- Install open-source ASIC toolchain on macOS/Linux
#
# Tools installed:
#   - Yosys          (synthesis)
#   - OpenSTA        (static timing analysis)
#   - OpenROAD       (place & route)
#   - Verilator      (RTL simulation / lint)
#   - cocotb         (Python-based RTL testbench)
#   - Magic          (DRC)
#   - KLayout        (DRC / GDS viewer)
#   - OpenLane 2     (full RTL-to-GDSII flow)
#   - SkyWater PDK   (sky130 open-source 130nm)
#
# Usage:
#   chmod +x scripts/install_toolchain.sh
#   ./scripts/install_toolchain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== coresmith ASIC Toolchain Installer ==="
echo "Project root: $PROJECT_ROOT"

OS="$(uname -s)"

# --------------------------------------------------------------------------
# 1. System package manager installs
# --------------------------------------------------------------------------
install_system_deps() {
    echo ""
    echo "--- Installing system dependencies ---"
    if [[ "$OS" == "Darwin" ]]; then
        if ! command -v brew &>/dev/null; then
            echo "ERROR: Homebrew not found. Install from https://brew.sh"
            exit 1
        fi
        brew install yosys verilator klayout python@3.11 || true
        echo "Note: 'magic' (VLSI DRC tool) is not in default Homebrew. Install from source or via Nix."
        echo "Note: OpenSTA and OpenROAD are best installed via OpenLane 2 (Nix or Docker)."
    elif [[ "$OS" == "Linux" ]]; then
        sudo apt-get update
        sudo apt-get install -y \
            yosys verilator magic klayout \
            build-essential cmake git python3 python3-pip python3-venv \
            tcl-dev tk-dev libffi-dev libssl-dev
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
}

# --------------------------------------------------------------------------
# 2. Python virtual environment for orchestrator
# --------------------------------------------------------------------------
setup_python_env() {
    echo ""
    echo "--- Setting up Python environment ---"
    VENV_DIR="$PROJECT_ROOT/orchestrator/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        python3 -m venv "$VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip
    pip install -e "$PROJECT_ROOT/orchestrator[dev]"
    pip install cocotb cocotb-bus cocotb-test
    pip install pytest-cocotb

    echo "Python env ready at $VENV_DIR"
}

# --------------------------------------------------------------------------
# 3. SkyWater PDK (sky130)
# --------------------------------------------------------------------------
install_sky130() {
    echo ""
    echo "--- Installing SkyWater Sky130 PDK ---"
    PDK_ROOT="${PDK_ROOT:-$PROJECT_ROOT/.pdk}"
    mkdir -p "$PDK_ROOT"

    if [[ ! -d "$PDK_ROOT/sky130A" ]]; then
        echo "Downloading sky130 PDK via volare..."
        pip install volare
        # Single source of truth for the PDK pin; bump in scripts/pdk-version.env.
        # If volare reports "not found remotely", check
        # `volare ls-remote --pdk sky130` for a current hash and update there.
        # shellcheck disable=SC1091
        source "$SCRIPT_DIR/pdk-version.env"
        volare enable --pdk sky130 --pdk-root "$PDK_ROOT" "$SKY130_PDK_COMMIT"
    else
        echo "Sky130 PDK already installed at $PDK_ROOT/sky130A"
    fi
    export PDK_ROOT
    echo "PDK_ROOT=$PDK_ROOT"
}

# --------------------------------------------------------------------------
# 4. OpenLane 2 (includes OpenROAD, OpenSTA, yosys, etc.)
# --------------------------------------------------------------------------
install_openlane() {
    echo ""
    echo "--- Installing OpenLane 2 ---"
    if ! command -v openlane &>/dev/null; then
        pip install openlane
        echo "OpenLane 2 installed via pip."
        echo "For full Nix-based install, see: https://openlane2.readthedocs.io"
    else
        echo "OpenLane 2 already available."
    fi
}

# --------------------------------------------------------------------------
# 5. RISC-V toolchain (optional, for firmware compilation)
# --------------------------------------------------------------------------
install_riscv_toolchain() {
    echo ""
    echo "--- RISC-V Toolchain (optional) ---"
    echo "Skipped by default. Install manually if needed:"
    echo "  macOS: brew install riscv-gnu-toolchain"
    echo "  Linux: sudo apt-get install gcc-riscv64-unknown-elf"
}

# --------------------------------------------------------------------------
# 6. Verify installations
# --------------------------------------------------------------------------
verify() {
    echo ""
    echo "--- Verifying installations ---"
    local tools=("yosys" "verilator" "magic" "python3")
    local missing=()
    for tool in "${tools[@]}"; do
        if command -v "$tool" &>/dev/null; then
            echo "  [OK] $tool: $(command -v "$tool")"
        else
            echo "  [MISSING] $tool"
            missing+=("$tool")
        fi
    done

    # README declares minimums: yosys >= 0.40, verilator >= 5.0. Ubuntu 22.04
    # apt ships 0.9 and 4.038 respectively, which silently break the pipeline.
    local stale=()
    if command -v yosys &>/dev/null; then
        local yv
        yv="$(yosys -V 2>&1 | grep -oE 'Yosys [0-9]+\.[0-9]+' | awk '{print $2}')"
        if [[ -n "$yv" ]] && awk -v v="$yv" 'BEGIN{exit !(v+0 < 0.40)}'; then
            echo "  [STALE] yosys $yv -- README requires >= 0.40 (apt jammy ships 0.9)"
            stale+=("yosys")
        fi
    fi
    if command -v verilator &>/dev/null; then
        local vv
        vv="$(verilator --version 2>&1 | grep -oE 'Verilator [0-9]+\.[0-9]+' | awk '{print $2}')"
        if [[ -n "$vv" ]] && awk -v v="$vv" 'BEGIN{exit !(v+0 < 5.0)}'; then
            echo "  [STALE] verilator $vv -- README requires >= 5.0 (apt jammy ships 4.038)"
            stale+=("verilator")
        fi
    fi

    # Check Python packages
    python3 -c "import cocotb; print(f'  [OK] cocotb {cocotb.__version__}')" 2>/dev/null || echo "  [MISSING] cocotb"
    python3 -c "import langgraph; print(f'  [OK] langgraph {langgraph.__version__}')" 2>/dev/null || echo "  [MISSING] langgraph"

    if [[ ${#missing[@]} -gt 0 ]] || [[ ${#stale[@]} -gt 0 ]]; then
        echo ""
        echo "WARNING: Toolchain is incomplete or below required versions."
        echo "         For a clean Linux install (no Nix, no Docker), use the"
        echo "         OSS-CAD-Suite tarball — see README \"Option C\"."
    else
        echo ""
        echo "All core tools verified."
    fi
}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
install_system_deps
setup_python_env
install_sky130
install_openlane
install_riscv_toolchain
verify

echo ""
echo "=== Toolchain setup complete ==="
echo "Activate the orchestrator venv with:"
echo "  source $PROJECT_ROOT/orchestrator/.venv/bin/activate"
