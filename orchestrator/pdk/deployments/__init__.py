# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in coresmith deployments.

Each module exposes a module-level ``DEPLOYMENT`` instance that the registry
(`orchestrator.pdk.registry`) resolves by name. A user's bring-your-own
deployment is a single ``.py`` file with the same ``DEPLOYMENT`` attribute,
loaded from ``CORESMITH_DEPLOYMENT=/abs/path/my_env.py``.
"""
