# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# coresmith -- AI-orchestrated ASIC pipeline
#
# This image bundles the full open-source EDA toolchain (Yosys, OpenROAD,
# Magic, netgen, KLayout, the Sky130 PDK) with the coresmith Python
# orchestrator and the Claude Code CLI, so the pipeline can run
# end-to-end inside a single container -- ideal for RunPod, EC2 or any
# CI environment that doesn't have Nix.
#
# Build:
#   docker build -t coresmith:latest .
#
# Run (interactive shell):
#   docker run --rm -it \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -v $(pwd)/.coresmith:/coresmith/.coresmith \
#     -v $(pwd)/rtl:/coresmith/rtl \
#     coresmith:latest
#
# Run (headless pipeline):
#   docker run --rm \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e CORESMITH_MODE=pipeline \
#     -v $(pwd)/.coresmith:/coresmith/.coresmith \
#     coresmith:latest

# -----------------------------------------------------------------------------
# Base: ghcr.io/efabless/openlane2 is a Nix-built image (~1.5 GB compressed)
# that bundles Yosys, OpenROAD, Magic, netgen, KLayout, Verilator, iverilog
# and OpenSTA already on PATH, plus a Python 3.11 environment. We layer
# Node + the Claude CLI on top via the image's bundled Nix.
#
# All efabless/openlane* images are Nix-only -- there is no Debian-based
# variant to fall back on. Earlier builds tried patching the glibc ELF
# (`@anthropic-ai/claude-code-linux-x64`); it segfaulted because the
# Bun-compiled binary's glibc ABI assumptions don't survive the move
# from FHS to nix-store paths even after patchelf. The musl variant
# (`@anthropic-ai/claude-code-linux-x64-musl`) is dynamically linked
# against /lib/ld-musl-x86_64.so.1, which on musl is also the libc --
# one symlink to nixpkgs.musl resolves the entire dependency, no
# patchelf needed.
# -----------------------------------------------------------------------------
FROM ghcr.io/efabless/openlane2:2.3.10 AS coresmith

RUN nix-channel --add https://nixos.org/channels/nixos-24.05 nixpkgs \
 && nix-channel --add https://nixos.org/channels/nixos-unstable nixos-unstable \
 && nix-channel --update \
 && nix-env -iA \
        nixos-unstable.nodejs_22 \
        nixpkgs.gnumake \
        nixpkgs.openssh \
        nixpkgs.musl \
        nixpkgs.gcc \
        nixpkgs.zlib \
 # Pull verilator from nixos-unstable so we get >= 5.036; the openlane2
 # base ships 5.018 which cocotb 2.0+ refuses ("cocotb requires
 # Verilator 5.036 or later"). Our PATH (set below) puts
 # /root/.nix-profile/bin before the base's verilator dir so this wins.
 # nixpkgs.gcc bundles gcc + g++; verilator shells out to `g++` to
 # compile the simulator binary and the openlane2 base does not put a
 # C++ compiler on PATH for non-yosys subprocesses.
 && nix-env -iA nixos-unstable.verilator

# nixpkgs.gcc's runtime libstdc++ lives under /root/.nix-profile/lib but
# isn't on the dynamic loader's default search path. PyPI's manylinux
# wheels for numpy 2.4+ are linked against libstdc++.so.6 at the standard
# FHS location -- without LD_LIBRARY_PATH, the wavekit sdist build fails
# its `import numpy as np` with
#   ImportError: libstdc++.so.6: cannot open shared object file
# Surface the nix-store libstdc++ at the loader's default search path.
ENV LD_LIBRARY_PATH="/root/.nix-profile/lib:${LD_LIBRARY_PATH}"

ENV PATH="/root/.nix-profile/bin:${PATH}"

# PyPI manylinux wheels (numpy 2.4+, used by wavekit's pyproject build env)
# dlopen libstdc++.so.6 via the loader's default FHS search paths. The
# openlane2 nix-only base has no /usr/lib/x86_64-linux-gnu, and
# nixpkgs.gcc's profile dir only contains the gcc *wrapper* -- libstdc++
# itself lives in the gcc-unwrapped output under /nix/store. Find it and
# symlink it into a path the loader actually searches.
#   build error this fixes:
#     ImportError: libstdc++.so.6: cannot open shared object file: No such file or directory
#
# The candidate must be picked by *library version*, not by store path.
# `find ... | sort -V | tail -1` sorts whole /nix/store paths, and those
# begin with a random 32-char hash -- so the "newest" library was really
# whichever hash happened to sort last. That silently selected nixos-24.05's
# gcc-13 libstdc++.so.6.0.32 (max CXXABI_1.3.14) even though the store also
# carries a newer one, and every consumer needing a later ABI then died:
#     node: /usr/lib/libstdc++.so.6: version `CXXABI_1.3.15' not found
#           (required by .../icu4c-78.3/lib/libicui18n.so.78)
# which broke `npm install -g` and failed the image build outright.
# Sorting on the *basename* orders by the real X.Y suffix. Newest is always
# the correct pick: libstdc++.so.6 and libz.so.1 are backward-compatible,
# so the highest version satisfies every older consumer too.
RUN set -eux \
 && mkdir -p /usr/lib /lib /etc/ld.so.conf.d \
 && for spec in \
        'libstdc++.so.6|libstdc\+\+\.so\.6\.[0-9]+\.[0-9]+' \
        'libz.so.1|libz\.so\.1\.[0-9]+\.[0-9]+' ; do \
        LIBNAME="${spec%%|*}"; PATTERN="${spec##*|}"; \
        LIB="$(find /nix/store -regextype posix-extended \
                -regex ".*/${PATTERN}$" -type f 2>/dev/null \
                | awk -F/ '{print $NF" "$0}' \
                | sort -V | tail -1 | cut -d' ' -f2-)"; \
        if [ -z "${LIB}" ] || [ ! -s "${LIB}" ]; then \
            echo "ERROR: could not locate ${LIBNAME} under /nix/store"; \
            exit 1; \
        fi; \
        echo "linking ${LIBNAME} -> ${LIB}"; \
        ln -sf "${LIB}" "/usr/lib/${LIBNAME}"; \
        ln -sf "${LIB}" "/lib/${LIBNAME}"; \
        dirname "${LIB}" >> /etc/ld.so.conf.d/nix-gcc.conf; \
    done \
 && sort -u -o /etc/ld.so.conf.d/nix-gcc.conf /etc/ld.so.conf.d/nix-gcc.conf \
 && (ldconfig 2>/dev/null || true) \
 && python3 -c "import ctypes; \
    [print(f'{n} resolves:', ctypes.CDLL(n)._name) for n in ('libstdc++.so.6','libz.so.1')]"

# pip's build-isolation venv runs subprocesses with a stripped env so
# the /etc/ld.so.cache fallback might not catch every loader. Export
# LD_LIBRARY_PATH at image level so the loader has at least one path
# guaranteed to be respected even in restricted subprocesses.
ENV LD_LIBRARY_PATH="/usr/lib:/lib:${LD_LIBRARY_PATH}"

# Verify the unstable-channel verilator is on PATH and >= 5.036
# (cocotb >= 2.0 requires it). Fails the build loud if PATH ordering
# accidentally resurfaces the openlane2 base's stale 5.018.
# Also verify g++ is reachable -- verilator shells out to it to
# compile the testbench simulator and the missing-compiler error
# only surfaces when sim runs, which is too late.
#
# `node --version` is checked here too: node is the first consumer of the
# libstdc++ symlinked above, and when that symlink points at too old a
# libstdc++ node cannot start at all. Catching it here reports the ABI
# mismatch directly instead of surfacing 4 layers later as an opaque
# `npm install` exit code 1.
RUN set -eux \
 && which verilator \
 && verilator --version \
 && verilator --version | python3 -c "import sys,re; v=sys.stdin.read().strip(); m=re.search(r'(\d+)\.(\d+)', v); maj,min=int(m.group(1)),int(m.group(2)); assert (maj,min) >= (5,36), f'verilator too old: {v}'; print(f'verilator OK: {v}')" \
 && which g++ \
 && g++ --version | head -1 \
 && which node \
 && node --version

# Make the musl ELF interpreter resolvable at the FHS path the binary
# is hard-linked against. nix's musl ships ld-musl-x86_64.so.1 which is
# also the libc; one symlink covers both. Without this, every musl
# binary on this image dies with the kernel's "required file not found".
RUN MUSL_LD="$(find /nix/store -maxdepth 4 -name 'ld-musl-x86_64.so.1' \
        \( -type f -o -type l \) 2>/dev/null | head -1)" \
 && test -n "${MUSL_LD}" \
 && mkdir -p /lib \
 && ln -sf "${MUSL_LD}" /lib/ld-musl-x86_64.so.1 \
 && echo "musl_ld=${MUSL_LD}"

# Same surgery for the glibc ELF interpreter. The GitHub Codespaces
# agent downloads a pre-built `vscode-remote-server` binary and execs
# it inside the container; the binary is dynamically linked against
# /lib64/ld-linux-x86-64.so.2 (standard FHS path). On the Nix-only
# openlane2 base, glibc lives at /nix/store/<hash>-glibc-*/lib/ but
# nothing at /lib64. Result: the agent's exec returns ENOENT and the
# caller sees the generic "failed to start vs code remote server" /
# "failed to start SSH server" -- because both `gh codespace ssh`
# and the web workbench go through the same agent-spawn path. Symlink
# the nix-store loader at the FHS path so the agent binary can run.
RUN GLIBC_LD="$(find /nix/store -path '*glibc-*/lib/ld-linux-x86-64.so.2' \
        \( -type f -o -type l \) 2>/dev/null | head -1)" \
 && test -n "${GLIBC_LD}" \
 && mkdir -p /lib64 \
 && ln -sf "${GLIBC_LD}" /lib64/ld-linux-x86-64.so.2 \
 && echo "glibc_ld=${GLIBC_LD}" \
 # Sanity check: a stock /bin/ls (which is dynamically linked) must
 # now resolve through this loader. If it doesn't, neither will the
 # codespaces agent.
 && /lib64/ld-linux-x86-64.so.2 --help 2>&1 | head -1

# Write a Debian-compatible /etc/os-release so the GitHub Codespaces
# agent's distro probe succeeds.  The devcontainer CLI runs
#   (cat /etc/os-release || cat /usr/lib/os-release) 2>/dev/null
# during boot to pick a matching vscode-remote-server binary (Debian
# / Ubuntu / Alpine variants are all distinct downloads). On the
# Nix-only openlane2 base neither file exists, so the probe returns
# exit code 1 with empty output; the agent then has no distro to
# target and the workbench dies with the generic
#   "failed to start vs code remote server"
# (which is also what `gh codespace ssh` rewrites to "failed to start
# SSH server"). The image is glibc-ABI compatible with Ubuntu after
# the loader symlinks above, so identifying as Ubuntu 22.04 is
# accurate enough for the agent's purposes.
RUN { \
        echo 'NAME="Ubuntu"'; \
        echo 'VERSION="22.04.5 LTS (Jammy Jellyfish)"'; \
        echo 'ID=ubuntu'; \
        echo 'ID_LIKE=debian'; \
        echo 'PRETTY_NAME="Ubuntu 22.04.5 LTS"'; \
        echo 'VERSION_ID="22.04"'; \
        echo 'HOME_URL="https://www.ubuntu.com/"'; \
        echo 'SUPPORT_URL="https://help.ubuntu.com/"'; \
        echo 'BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"'; \
        echo 'PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"'; \
        echo 'VERSION_CODENAME=jammy'; \
        echo 'UBUNTU_CODENAME=jammy'; \
    } > /etc/os-release \
 && mkdir -p /usr/lib && cp /etc/os-release /usr/lib/os-release \
 # Sanity check: the probe the agent runs must now succeed.
 && (cat /etc/os-release || cat /usr/lib/os-release) >/dev/null 2>&1

# Nix-built Node sets npm's default prefix into the read-only Nix store,
# so `npm install -g` "succeeds" but the bin can't symlink anywhere on
# PATH. Pin the prefix to a writable dir we explicitly put on PATH.
ENV NPM_CONFIG_PREFIX=/opt/npm-global
ENV PATH="/opt/npm-global/bin:${PATH}"

# Install the wrapper package + the Linux-x64-musl native variant.
# `--libc=musl` lies to npm's resolver about the host libc so it
# accepts the musl subpackage (Nix Node is glibc-built, so npm would
# otherwise reject it with EBADPLATFORM). `--force` is belt-and-braces
# in case the resolver still complains. The wrapper's install.cjs
# postinstall will still pick the glibc variant for bin/claude.exe
# because it reads process.report.glibcVersionRuntime directly -- the
# RUN below replaces that with a symlink to the musl binary.
RUN mkdir -p /opt/npm-global \
 && npm install -g --force --libc=musl \
        @moonshot-ai/kimi-code \
        @anthropic-ai/claude-code \
        @anthropic-ai/claude-code-linux-x64-musl \
        opencode-ai \
        opencode-linux-x64-musl

# `npm install -g`'s postinstall on the wrapper package may have copied
# the glibc binary over bin/claude.exe (because Nix Node is glibc-
# built). Force-replace bin/claude with a symlink to the musl-variant's
# native binary so we use the version that actually runs on this image.
RUN set -eux \
 && MUSL_BIN=/opt/npm-global/lib/node_modules/@anthropic-ai/claude-code-linux-x64-musl/claude \
 && test -x "${MUSL_BIN}" \
 && ln -sf "${MUSL_BIN}" /opt/npm-global/bin/claude

# OpenCode also ships glibc and musl native variants. This image needs the
# explicit x64-musl binary for the same reason as Claude Code above.
RUN set -eux \
 && OPENCODE_MUSL_BIN=/opt/npm-global/lib/node_modules/opencode-linux-x64-musl/bin/opencode \
 && test -x "${OPENCODE_MUSL_BIN}" \
 && ln -sf "${OPENCODE_MUSL_BIN}" /opt/npm-global/bin/opencode

# Capture the resolved agent CLI paths at build time and bake them so runtime
# resolution can't drift if PATH changes under us. Fail the build loud if any
# CLI is unusable -- the runtime "PermissionError: ''" failure mode is much
# harder to debug.
RUN set -eux \
 && CLAUDE_BIN="$(command -v claude)" \
 && OPENCODE_BIN="$(command -v opencode)" \
 && KIMI_BIN="$(command -v kimi)" \
 && test -x "${CLAUDE_BIN}" \
 && test -x "${OPENCODE_BIN}" \
 && test -x "${KIMI_BIN}" \
 && claude --version \
 && opencode --version \
 && kimi --version \
 && printf 'CLAUDE_CLI_PATH=%s\nOPENCODE_CLI_PATH=%s\nKIMI_CLI_PATH=%s\n' \
      "${CLAUDE_BIN}" "${OPENCODE_BIN}" "${KIMI_BIN}" > /etc/coresmith.env

# sshd setup so RunPod / interactive users can ssh in (and the web
# terminal works because PID 1 stays alive even after the pipeline
# exits -- see runpod_entrypoint.sh's pipeline keep-alive). Host keys
# are baked into the image; per-deploy authorized_keys is written by
# the entrypoint from the PUBLIC_KEY env var.
#
# Three quirks of the openlane2 nix-only base we work around here:
#   1. No `sshd` privsep user exists -- nix's openssh refuses to start
#      without one ("Privilege separation user sshd does not exist").
#      We add it via useradd; UsePrivilegeSeparation=no would also work
#      but is deprecated in modern OpenSSH.
#   2. No /bin/bash -- the openlane2 base ships only nix-store paths,
#      so RunPod's `docker exec /bin/bash` (used by their SSH proxy
#      and web terminal) fails. Symlink to the nix-store bash.
#   3. /var/empty (the privsep chroot) must exist and be root-owned.
#   4. No /usr/sbin/sshd -- the GitHub Codespaces agent does its
#      "is sshd installed?" check by stat'ing standard FHS paths
#      (/usr/sbin/sshd, /usr/bin/ssh-keygen). Without these symlinks
#      it logs "Please check if an SSH server is installed" and
#      `gh codespace ssh` is dead even though sshd is on $PATH.
#      Same workaround as #1/#2: symlink the nix-store binary at
#      the FHS path the consumer expects.
RUN BASH_BIN="$(command -v bash)" \
 && SSHD_BIN="$(command -v sshd)" \
 && SSH_KEYGEN_BIN="$(command -v ssh-keygen)" \
 && test -x "${BASH_BIN}" \
 && test -x "${SSHD_BIN}" \
 && test -x "${SSH_KEYGEN_BIN}" \
 && ln -sf "${BASH_BIN}" /bin/bash \
 && mkdir -p /usr/sbin /usr/bin \
 && ln -sf "${SSHD_BIN}" /usr/sbin/sshd \
 && ln -sf "${SSH_KEYGEN_BIN}" /usr/bin/ssh-keygen \
 && if ! getent passwd sshd >/dev/null 2>&1; then \
        if command -v useradd >/dev/null 2>&1; then \
            useradd -r -M -d /var/empty -s /usr/sbin/nologin sshd; \
        else \
            echo 'sshd:x:74:74:Privilege-separated SSH:/var/empty:/usr/sbin/nologin' >> /etc/passwd \
         && echo 'sshd:x:74:' >> /etc/group; \
        fi; \
    fi \
 && mkdir -p /etc/ssh /var/run/sshd /run/sshd /root/.ssh /var/empty \
 && chmod 700 /root/.ssh \
 && chmod 755 /var/empty \
 && ssh-keygen -A \
 # The openlane2 base leaves root with `!` in /etc/shadow (locked
 # password placeholder). nix-built openssh checks shadow even with
 # `PermitRootLogin without-password` and rejects pubkey auth as
 # "User root not allowed because account is locked". Replace `!` with
 # `*` (no-password, but not locked) so pubkey auth is permitted.
 # We use python because sed/passwd/usermod aren't on the base PATH.
 && python3 -c "p='/etc/shadow'; t=open(p).read(); open(p,'w').write(t.replace('root:!:','root:*:',1))" \
 # Locate sftp-server from the nix openssh package so SCP/SFTP work
 # over our sshd (`Subsystem sftp` is required; otherwise scp gets
 # "subsystem request failed"). Symlink for sshd_config stability.
 && SFTP_SERVER="$(find /nix/store -path '*openssh*/libexec/sftp-server' -type f 2>/dev/null | head -1)" \
 && test -x "${SFTP_SERVER}" \
 && { \
        echo "Port 22"; \
        echo "PermitRootLogin prohibit-password"; \
        echo "PasswordAuthentication no"; \
        echo "PubkeyAuthentication yes"; \
        echo "AuthorizedKeysFile /root/.ssh/authorized_keys"; \
        echo "ChallengeResponseAuthentication no"; \
        echo "UsePAM no"; \
        echo "PrintMotd no"; \
        echo "AcceptEnv LANG LC_*"; \
        echo "Subsystem sftp ${SFTP_SERVER}"; \
    } > /etc/ssh/sshd_config

# -----------------------------------------------------------------------------
# Python venv + coresmith deps. Done in two layers so a code-only edit
# doesn't bust the dependency cache. The base image's python3 is 3.11.9 so
# we use it directly (no deadsnakes / system Python dance).
# -----------------------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/coresmith-venv
RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /coresmith

COPY requirements.txt ./
COPY orchestrator/pyproject.toml ./orchestrator/pyproject.toml
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install cocotb cocotb-bus

COPY . /coresmith
RUN pip install -e ./orchestrator

# -----------------------------------------------------------------------------
# Sky130 PDK -- install at build time so first pipeline run isn't a 5GB
# download. The pin lives in scripts/pdk-version.env so a bump only needs
# to land in one place (Dockerfile + install_toolchain.sh + CI all read it).
# -----------------------------------------------------------------------------
ENV PDK_ROOT=/coresmith/.pdk
RUN pip install volare \
 && . /coresmith/scripts/pdk-version.env \
 && volare enable \
        --pdk sky130 \
        --pdk-root "${PDK_ROOT}" \
        "${SKY130_PDK_COMMIT}"

# -----------------------------------------------------------------------------
# Tool wrappers: openlane2 already exposes yosys / openroad / magic / netgen
# / klayout / verilator at the bare names on $PATH, so the scripts/*-nix.sh
# wrappers just need to exec the real binary. We override config.yaml to
# point at bare names.
# -----------------------------------------------------------------------------
RUN python3 -c "import yaml, pathlib; \
p = pathlib.Path('orchestrator/config.yaml'); \
c = yaml.safe_load(p.read_text()); \
c['backend']['openroad_binary'] = 'openroad'; \
c['backend']['magic_binary']    = 'magic'; \
c['backend']['netgen_binary']   = 'netgen'; \
c['backend']['yosys_binary']    = 'yosys'; \
c['backend']['klayout_binary']  = 'klayout'; \
p.write_text(yaml.safe_dump(c, sort_keys=False))"

# Make the .coresmith / rtl / tb / arch dirs that the pipeline writes to
# world-writable so they survive `docker run --user $(id -u)` style invocations.
RUN mkdir -p /coresmith/.coresmith /coresmith/rtl /coresmith/tb /coresmith/arch \
             /coresmith/syn /coresmith/sim_build /coresmith/pnr \
 && chmod -R 0777 /coresmith/.coresmith /coresmith/rtl /coresmith/tb /coresmith/arch \
                  /coresmith/syn /coresmith/sim_build /coresmith/pnr

# Default behaviour: drop into a shell. Set CORESMITH_MODE=pipeline (or
# =mcp, =backend) to launch a specific entry point via the entrypoint.
ENV CORESMITH_MODE=shell
ENV CORESMITH_PROJECT_ROOT=/coresmith

ENTRYPOINT ["/coresmith/scripts/runpod_entrypoint.sh"]
CMD []
