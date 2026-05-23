#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# openroad-nix.sh -- Run OpenROAD inside a Nix shell
#
# Usage:
#   scripts/openroad-nix.sh -version
#   scripts/openroad-nix.sh -no_init script.tcl
#   scripts/openroad-nix.sh < script.tcl
#
# The coresmith backend graph calls this instead of a bare "openroad" binary.
# Configure the path in orchestrator/config.yaml under backend.openroad_binary.

set -euo pipefail

exec nix shell "nixpkgs#openroad" --command openroad -exit "$@"
