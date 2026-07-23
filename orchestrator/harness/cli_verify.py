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
from typing import Any

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


def _scoreboard(root: Path):
    try:
        from orchestrator.state_store.store import Scoreboard
        return Scoreboard(root)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def cmd_verify_model(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    from orchestrator.harness import blocks as B

    if getattr(args, "all", False):
        names = B.block_names(root)
        if not names:
            print("no blocks resolved", file=sys.stderr)
            return EXIT_USAGE
        results = {}
        codes = []
        for name in names:
            r = V.verify_model(root, name, skip_size=getattr(args, "skip_size", False))
            results[name] = r.to_json()
            codes.append(r.exit_code)
            if not getattr(args, "json", False):
                print(f"{name:<24} {r.to_human().splitlines()[0]}")
        if getattr(args, "json", False):
            print(json.dumps(results, indent=2, default=str))
        # any real failure wins; else infra; else pass; else all-skipped.
        if EXIT_FAIL in codes:
            return EXIT_FAIL
        if EXIT_INFRA in codes:
            return EXIT_INFRA
        if any(c == EXIT_PASS for c in codes):
            return EXIT_PASS
        return EXIT_SKIP

    if not getattr(args, "block", None):
        print("verify model needs a <block> or --all", file=sys.stderr)
        return EXIT_USAGE
    result = V.verify_model(root, args.block, skip_size=getattr(args, "skip_size", False))
    return _emit_result(args, result)


def cmd_verify_chip_model(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    return _emit_result(args, V.verify_chip_model(root))


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
    result = V.verify_synth(
        root, spec,
        full=getattr(args, "full", False),
        timeout_s=getattr(args, "timeout", 300),
        scoreboard=_scoreboard(root),
        record_source="agent",
    )
    return _emit_result(args, result)


def cmd_verify_equiv(args) -> int:
    """Practice tool: the deterministic RTL<->model byte-exact equivalence check
    on ONE block with a VISIBLE, reproducible dev seed.

    This is the SAME engine-authored check the pipeline runs as a gate -- but the
    gate draws a fresh HIDDEN seed post-freeze while this PRINTS the seed so you
    can reproduce a failure. Pass a block on several random dev seeds before you
    submit; the gate is one more draw your practice never saw. Overfitting to one
    dev seed fails the gate; generalising passes both.
    """
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
    from orchestrator.harness.seed_provider import dev_seed

    spec = _require_block_spec(root, args.block) or {"name": args.block}
    name = spec.get("name", args.block)
    rtl_path = V._resolve_rtl_path(root, spec)
    if not Path(rtl_path).exists():
        print(f"RTL not found: {rtl_path}", file=sys.stderr)
        return EXIT_INFRA
    seed = dev_seed(getattr(args, "seed", None))
    eq = V.run_block_equiv_gate(name, rtl_path, root, seed=seed)
    payload = {"block": name, "seed": seed, **eq}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        status = ("PASS" if eq.get("passed") else
                  "FAIL" if eq.get("failed_closed") or
                  (eq.get("ran") and not eq.get("skipped")) else "SKIP")
        print(f"equiv {name}: {status}  (seed={seed}, "
              f"vectors={eq.get('checked_vectors')})")
        if eq.get("reason"):
            print(f"  reason: {eq.get('reason')}")
        print(f"  reproduce: coresmith verify equiv {name} --seed {seed}")
    if eq.get("passed"):
        return EXIT_PASS
    if eq.get("failed_closed"):
        return EXIT_FAIL
    if eq.get("skipped"):
        return EXIT_SKIP
    if eq.get("ran"):
        return EXIT_FAIL  # ran, no match, not an honest skip => real mismatch
    return EXIT_SKIP  # gate not applicable (no golden / equiv disabled)


def cmd_verify_chip(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.harness import verify as V
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

    vm = vsub.add_parser("model", help="elaborate + interface + size a block model")
    add_project_root(vm)
    add_json(vm)
    vm.add_argument("block", nargs="?", help="block name (omit with --all)")
    vm.add_argument("--all", action="store_true", help="verify every block model")
    vm.add_argument("--skip-size", action="store_true", dest="skip_size")
    vm.set_defaults(func=run_wrap(cmd_verify_model))

    vcm = vsub.add_parser("chip-model", help="composed model vs golden byte-exact")
    add_project_root(vcm)
    add_json(vcm)
    vcm.set_defaults(func=run_wrap(cmd_verify_chip_model))

    vr = vsub.add_parser("rtl", help="lint + sim + RTL/model equivalence")
    add_project_root(vr)
    add_json(vr)
    vr.add_argument("block")
    vr.add_argument("--seed", type=int, default=None)
    vr.add_argument("--tb", default=None, help="override testbench path")
    vr.add_argument("--no-equiv", action="store_true", dest="no_equiv")
    vr.add_argument("--coverage", action="store_true")
    vr.add_argument("--lint-only", action="store_true", dest="lint_only")
    vr.set_defaults(func=run_wrap(cmd_verify_rtl))

    vs = vsub.add_parser("synth", help="synthesizability / PPA probe")
    add_project_root(vs)
    add_json(vs)
    vs.add_argument("block")
    vs.add_argument("--full", action="store_true", help="run synthesize_block (PDK)")
    vs.add_argument("--timeout", type=int, default=300, help="probe timeout (s)")
    vs.set_defaults(func=run_wrap(cmd_verify_synth))

    ve = vsub.add_parser(
        "equiv",
        help="RTL<->model byte-exact equivalence, VISIBLE dev seed (practice tool)")
    add_project_root(ve)
    add_json(ve)
    ve.add_argument("block")
    ve.add_argument("--seed", type=int, default=None,
                    help="reproduce a specific case; default = fresh printed seed")
    ve.set_defaults(func=run_wrap(cmd_verify_equiv))

    vc = vsub.add_parser("chip", help="integrated chip_top DV")
    add_project_root(vc)
    add_json(vc)
    vc.add_argument("--stimulus", default=None)
    vc.add_argument("--seed", type=int, default=None)
    vc.add_argument("--tb", default=None)
    vc.set_defaults(func=run_wrap(cmd_verify_chip))
