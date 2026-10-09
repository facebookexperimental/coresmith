# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith verify ...`` subcommands.

Kept separate from ``harness.cli`` (which stays langgraph-free at import) so the
verify handlers -- which DO reach into langgraph -- only load their heavy
dependencies inside the handler bodies at invocation time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Exit codes (kept in sync with harness.cli).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3
EXIT_SKIP = 4


def _bootstrap(args) -> Path:
    from orchestrator.harness.env import bootstrap_project_root
    return bootstrap_project_root(getattr(args, "project_root", None))


def _emit_result(args, result) -> int:
    if getattr(args, "json", False):
        print(json.dumps(result.to_json(), indent=2, default=str))
    else:
        print(result.to_human())
    return result.exit_code


def _engine_logs_to_stderr():
    """The engine helpers print progress lines (``[SIM] timeout=...``) to
    stdout; during a verify call they go to stderr so ``--json`` stdout is
    exactly one document (the same context ``coresmith model`` uses)."""
    from orchestrator.harness.cli import _engine_logs_to_stderr as _ctx
    return _ctx()


def _scoreboard(root: Path):
    try:
        from orchestrator.state_store.store import Scoreboard
        return Scoreboard(root)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _require_block_spec(root: Path, name: str):
    from orchestrator.harness import blocks as B
    return B.load_block_spec(root, name)


def cmd_verify_rtl(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    spec = _require_block_spec(root, args.block)
    if spec is None:
        # Fall back to a minimal spec so a bare `verify rtl <block>` still works
        # against conventional rtl/<block>.v + tb/cocotb/test_<block>.py.
        spec = {"name": args.block}
    with _engine_logs_to_stderr():
        result = V.verify_rtl(
            root, spec,
            seed=getattr(args, "seed", None),
            tb_path=getattr(args, "tb", None),
            no_equiv=getattr(args, "no_equiv", False),
            lint_only=getattr(args, "lint_only", False),
            coverage=getattr(args, "coverage", False),
            record_source="agent",
            scoreboard=_scoreboard(root),
        )
    return _emit_result(args, result)


def cmd_verify_synth(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    spec = _require_block_spec(root, args.block) or {"name": args.block}
    with _engine_logs_to_stderr():
        result = V.verify_synth(
            root, spec,
            full=getattr(args, "full", False),
            timeout_s=getattr(args, "timeout", 300),
            scoreboard=_scoreboard(root),
            record_source="agent",
        )
    return _emit_result(args, result)


def cmd_verify_chip(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    with _engine_logs_to_stderr():
        result = V.verify_chip(
            root,
            tb_path=getattr(args, "tb", None),
            seed=getattr(args, "seed", None),
            stimulus=getattr(args, "stimulus", None),
            scoreboard=_scoreboard(root),
            record_source="agent",
        )
    return _emit_result(args, result)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def register_verify(sub, run_wrap, add_project_root, add_json) -> None:
    """Add the ``verify`` command tree to the CLI subparser action."""
    vp = sub.add_parser("verify", help="re-run the engine DV/synth harness")
    vsub = vp.add_subparsers(dest="verify_cmd", required=True)

    vr = vsub.add_parser("rtl", help="lint + sim")
    add_project_root(vr)
    add_json(vr)
    vr.add_argument("block")
    vr.add_argument("--seed", type=int, default=None)
    vr.add_argument("--tb", default=None, help="override testbench path")
    vr.add_argument("--coverage", action="store_true")
    vr.add_argument("--lint-only", action="store_true", dest="lint_only")
    vr.set_defaults(func=run_wrap(cmd_verify_rtl))

    vs = vsub.add_parser("synth", help="synthesizability / PPA probe")
    add_project_root(vs)
    add_json(vs)
    vs.add_argument("block")
    vs.add_argument("--full", action="store_true", help=("run synthesize_block (PDK) + pre-layout STA; no WNS is a tool error "
                    "(exit 3) unless CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED=1"))
    vs.add_argument("--timeout", type=int, default=300, help="probe timeout (s)")
    vs.set_defaults(func=run_wrap(cmd_verify_synth))

    vc = vsub.add_parser("chip", help="integrated chip_top DV")
    add_project_root(vc)
    add_json(vc)
    vc.add_argument("--stimulus", default=None)
    vc.add_argument("--seed", type=int, default=None)
    vc.add_argument("--tb", default=None)
    vc.set_defaults(func=run_wrap(cmd_verify_chip))
