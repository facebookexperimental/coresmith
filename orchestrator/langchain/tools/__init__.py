# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""(Historical) LangChain EDA tool wrappers.

The ``.eda_tools`` module this package used to import (``CocotbRunTool``,
``YosysLintTool``, ...) was deleted long ago, leaving the package import-broken
and unused. EDA tools now live behind named verbs in the deployment layer --
see ``orchestrator/pdk/base.py`` (the :class:`EdaTool` ABC) and
``orchestrator/pdk/deployments/`` (the concrete tool classes).

Kept as an empty, importable package so any stale reference degrades to an empty
namespace rather than an ImportError.
"""

__all__: list[str] = []
