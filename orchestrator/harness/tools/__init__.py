# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The architect's tools (langgraph-free): what the architecture graph nodes
used to do deterministically, exposed as functions the CLI, the daemon and
the graph all call. ``register`` parses + validates + records an artifact,
``stages`` is the state machine."""
