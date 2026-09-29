# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Process liveness helpers shared by the state store and the daemon.

``ProjectDB`` must stay import-light (no langgraph, no daemon), so the pid
check the lifecycle used for daemon.json lives here instead of in
``graph_lifecycle``.
"""
from __future__ import annotations

import os
import socket


def pid_is_alive(pid: object) -> bool:
    """Whether ``pid`` names a live process on THIS host."""
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "localhost"
