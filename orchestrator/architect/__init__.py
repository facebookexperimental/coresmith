# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native sessions driven through the coresmith CLI (cluster workers only).

The Architect -- the coding agent that talks to the human and calls the CLI
-- is outside the engine; nothing here launches, resumes or decides for it.
``ArchitectSession`` is the historical name of the session base class."""
from .session import ArchitectSession

__all__ = ["ArchitectSession"]
