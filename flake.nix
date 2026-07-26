# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# coresmith -- AI-orchestrated ASIC pipeline (local-dev Nix flake)
#
# Pins nixpkgs so every developer / CI run gets the same versions of
# Yosys, OpenROAD, Magic, netgen, KLayout, Verilator, etc. Drops the
# need for individual nix-shell invocations from the scripts/*-nix.sh
# wrappers -- run everything inside a single `nix develop` shell.
#
# Usage:
#   nix develop                # interactive devShell with all tools on PATH
#   nix develop -c make pipeline
#   nix develop -c yosys -V
#
# First-run download is a few GB (one nixpkgs closure for the whole
# toolchain); subsequent invocations are instant.
#
# Requires: a Nix install with flakes enabled. Add this once to
# ~/.config/nix/nix.conf:
#   experimental-features = nix-command flakes

{
  description = "coresmith -- AI-orchestrated ASIC pipeline";

  inputs = {
    # Pinned to a known-good nixos-24.05 channel commit. Bump after
    # validating the flow on a newer nixpkgs (CI: `nix flake update`
    # then run `make test` and the `requires_nix`-marked tests).
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Python environment with the orchestrator dependencies. We pin
        # versions through the existing requirements.txt so the flake
        # and the pip install path agree.
        py = pkgs.python311;

        # EDA tools available in nixpkgs. magic is named magic-vlsi to
        # disambiguate from the GUI photo viewer of the same name.
        edaTools = with pkgs; [
          yosys
          verilator
          openroad
          magic-vlsi
          netgen
          klayout
        ];

        # Build / dev tooling needed by the orchestrator and tests.
        devTools = with pkgs; [
          py
          py.pkgs.pip
          py.pkgs.virtualenv
          gnumake
          git
          nodejs_20      # for Claude Code and OpenCode (`npm install -g opencode-ai`)
        ];
      in {
        # Default devShell: `nix develop`
        devShells.default = pkgs.mkShell {
          name = "coresmith-dev";
          packages = edaTools ++ devTools;

          # Inside the shell, the bare tool names work, so the
          # scripts/*-nix.sh wrappers become redundant. We rewrite the
          # backend tool config on shell entry to avoid the redundant
          # `nix shell` re-entry from inside `nix develop`.
          shellHook = ''
            export CORESMITH_PROJECT_ROOT="''${CORESMITH_PROJECT_ROOT:-$PWD}"
            export PDK_ROOT="''${PDK_ROOT:-$CORESMITH_PROJECT_ROOT/.pdk}"

            # Point the orchestrator at the bare binaries on $PATH instead
            # of re-invoking `nix shell` for every EDA call. Leaves the
            # checked-in config.yaml unchanged on disk; the override is
            # via env vars consumed by backend_helpers._resolve_tool.
            export CORESMITH_BACKEND_OPENROAD=$(command -v openroad)
            export CORESMITH_BACKEND_MAGIC=$(command -v magic)
            export CORESMITH_BACKEND_NETGEN=$(command -v netgen)
            export CORESMITH_BACKEND_YOSYS=$(command -v yosys)
            export CORESMITH_BACKEND_KLAYOUT=$(command -v klayout)

            echo "[coresmith] devShell ready."
            echo "  yosys     = $(command -v yosys)"
            echo "  openroad  = $(command -v openroad)"
            echo "  verilator = $(command -v verilator)"
            echo "  PDK_ROOT  = $PDK_ROOT"
            echo
            echo "Next: python -m venv venv && source venv/bin/activate &&"
            echo "      pip install -r requirements.txt && pip install -e orchestrator/"
            echo "      claude login   # default provider; if not already authenticated"
            echo "      npm install -g opencode-ai  # for CORESMITH_LLM_PROVIDER=opencode"
            echo "      make pipeline  # or: make mcp"
          '';
        };

        # `nix flake check` -- just confirms the devShell builds for
        # this system. Real test coverage lives in `make test`.
        checks.devShell = self.devShells.${system}.default;
      });
}
