# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith verify ...`` + read-only state queries.

``register_subcommands(sub)`` wires the harness subcommands onto the existing
``bin/coresmith`` argparse tree. Uniform exit codes across every subcommand::

    0  pass
    1  fail
    2  usage / unknown block
    3  infra / timeout
    4  skip / cannot-judge

``--json`` is accepted on all subcommands (machine-readable output).

CRITICAL: this module MUST import without importing ``orchestrator.langgraph``
(``pipeline_helpers.PROJECT_ROOT`` freezes at import). Every heavy import is
therefore deferred into the handler bodies -- keep it that way (there is a unit
test asserting langgraph is not imported by importing this module).
"""

from __future__ import annotations

import json
import sys

# Exit codes (single source of truth).
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INFRA = 3
EXIT_SKIP = 4


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _emit(args, payload: dict, human: str) -> None:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(human)


def _bootstrap(args):
    from orchestrator.harness.env import bootstrap_project_root
    return bootstrap_project_root(getattr(args, "project_root", None))


# ---------------------------------------------------------------------------
# Read-only queries (direct disk reads; scoreboard when present, else fallback)
# ---------------------------------------------------------------------------
def cmd_dv_status(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = getattr(args, "block", None)
    sb = Scoreboard(root)

    if sb.exists():
        rows = sb.latest_dv(block=block)
        for r in rows:
            bp = root / ".coresmith" / "blocks" / str(r.get("block")) / "best_result.json"
            try:
                r["stale"] = bp.exists() and bp.stat().st_mtime > (r.get("ts") or 0)
            except OSError:
                r["stale"] = False
        payload = {"source": "scoreboard", "block": block, "rows": rows}
        human_lines = [
            f"{r.get('block'):<22} {r.get('scope'):<11} "
            f"{'PASS' if r.get('passed') else ('SKIP' if r.get('skipped') else 'FAIL')} "
            f"src={r.get('source')} attempt={r.get('attempt')} "
            f"tests={r.get('tests_passed')}/{r.get('tests_total')}"
            + ("  [stale]" if r.get("stale") else "")
            for r in rows
        ] or ["(no dv rows recorded)"]
        _emit(args, payload, "\n".join(human_lines))
        return EXIT_PASS

    # Disk fallback: blocks/<b>/best_result.json
    rows = []
    blocks_dir = root / ".coresmith" / "blocks"
    names = [block] if block else sorted(
        p.name for p in blocks_dir.glob("*") if p.is_dir()
    ) if blocks_dir.is_dir() else []
    for name in names:
        bp = blocks_dir / name / "best_result.json"
        if bp.exists():
            try:
                data = json.loads(bp.read_text())
            except Exception:  # noqa: BLE001
                continue
            rows.append({
                "block": name, "scope": "rtl", "source": "disk",
                "passed": bool(data.get("sim_passed")),
                "attempt": data.get("attempt"),
                "tests_passed": data.get("tests_passed"),
                "tests_total": data.get("tests_total"),
            })
    payload = {"source": "disk", "block": block, "rows": rows}
    human = "\n".join(
        f"{r['block']:<22} rtl        "
        f"{'PASS' if r['passed'] else 'FAIL'} (disk best_result.json) "
        f"tests={r.get('tests_passed')}/{r.get('tests_total')}"
        for r in rows
    ) or "(no scoreboard.db and no best_result.json found)"
    _emit(args, payload, human)
    return EXIT_PASS


def cmd_ppa(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = args.block
    sb = Scoreboard(root)
    history = bool(getattr(args, "history", False))

    if sb.exists() and (sb.latest_ppa(block) or sb.ppa_rows(block)):
        rows = sb.ppa_rows(block) if history else [sb.latest_ppa(block)]
        rows = [r for r in rows if r]
        payload = {"source": "scoreboard", "block": block, "rows": rows}
        human = "\n".join(
            f"[{r.get('probe')}] ff={r.get('ff')} cells={r.get('cells')} "
            f"mem_bits={r.get('mem_bits')} area={r.get('area_um2')} "
            f"wns={r.get('wns_ns')} elaborated={r.get('elaborated')} "
            f"ppa_ok={r.get('ppa_ok')} (attempt={r.get('attempt')}, {r.get('source')})"
            for r in rows
        ) or "(no ppa rows)"
        _emit(args, payload, human)
        return EXIT_PASS

    # Disk fallback: parse syn/output/<b>/<b>_report.txt for FF count.
    report = root / "syn" / "output" / block / f"{block}_report.txt"
    if report.exists():
        try:
            from orchestrator.langgraph.ppa_check import count_flops_from_stat
            text = report.read_text(errors="replace")
            ff = count_flops_from_stat(text)
            payload = {
                "source": "disk", "block": block, "ff": ff,
                "report_path": str(report),
            }
            _emit(args, payload, f"{block}: ff={ff} (from {report.name})")
            return EXIT_PASS
        except Exception as exc:  # noqa: BLE001
            print(f"could not parse {report}: {exc}", file=sys.stderr)
            return EXIT_INFRA

    payload = {"source": "none", "block": block, "reason": "no ppa data"}
    _emit(args, payload, f"{block}: no PPA data (no scoreboard row, no synth report)")
    return EXIT_SKIP


def cmd_coverage(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    from orchestrator.state_store.store import Scoreboard

    block = args.block
    sb = Scoreboard(root)
    row = sb.coverage_latest(block) if sb.exists() else None
    if not row:
        payload = {"source": "none", "block": block, "reason": "no coverage recorded"}
        _emit(
            args, payload,
            f"{block}: no coverage recorded "
            "(re-run `coresmith verify rtl <block> --coverage`)",
        )
        return EXIT_SKIP

    uncovered = []
    try:
        uncovered = json.loads(row.get("uncovered") or "[]")
    except Exception:  # noqa: BLE001
        uncovered = []
    payload = {
        "source": "scoreboard", "block": block,
        "points_total": row.get("points_total"),
        "points_hit": row.get("points_hit"),
        "pct": row.get("pct"),
    }
    if getattr(args, "uncovered", False):
        payload["uncovered"] = uncovered
    human = (
        f"{block}: {row.get('points_hit')}/{row.get('points_total')} "
        f"points ({row.get('pct')}%)"
    )
    if getattr(args, "uncovered", False):
        human += "\nUncovered:\n" + "\n".join(
            f"  {u.get('file')}:{u.get('line')}  {u.get('text')}"
            for u in uncovered[:200]
        )
    _emit(args, payload, human)
    return EXIT_PASS


def cmd_contracts(args) -> int:
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    block = args.block
    try:
        from orchestrator.langchain.agents.contract_lookup import load_block_contracts
        view = load_block_contracts(str(root), block)
    except Exception as exc:  # noqa: BLE001
        print(f"contract lookup failed: {exc}", file=sys.stderr)
        return EXIT_INFRA
    edges = view.get("edges") or []
    payload = {"block": block, "defaults": view.get("defaults") or {}, "edges": edges}
    human_lines = []
    if view.get("defaults"):
        human_lines.append(f"defaults: {view['defaults']}")
    for e in edges:
        human_lines.append(
            f"[{e.get('role')}] {e.get('producer_block')} -> "
            f"{e.get('consumer_block')} ({e.get('signal') or e.get('name') or '?'})"
        )
    _emit(args, payload, "\n".join(human_lines) or f"{block}: no contract edges")
    return EXIT_PASS


def cmd_golden_check(args) -> int:
    """Practice tool: probe a block's generated golden/model for degeneracy.

    Runs the SAME check `_maybe_generate_block_golden` runs to close the swallow:
    the model imports + defines its `@block`, the golden reference resolves, and
    (for free-function goldens) the block's golden slice is exercised on a
    stimulus. Honest-SKIPs where it can't conclude (method-based goldens, no
    slice) rather than false-failing.
    """
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    try:
        # Lazy import: model_integration lives under orchestrator.architecture
        # (not langgraph), keeping this module langgraph-free at import.
        from orchestrator.architecture.model_integration import (
            check_golden_feasibility,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"golden-check unavailable: {exc}", file=sys.stderr)
        return EXIT_INFRA
    res = check_golden_feasibility(str(root), args.block)
    ran, passed, skipped = res.get("ran"), res.get("passed"), res.get("skipped")
    status = ("SKIP" if skipped or not ran else "PASS" if passed else "FAIL")
    reach = (res.get("checks", {}) or {}).get("slice_reachability", {}) or {}
    human = f"golden-check {args.block}: {status}  ({res.get('reason') or 'ok'})"
    if reach:
        human += (f"\n  slice reachability: {reach.get('verdict')} -- "
                  f"{reach.get('reason', '')}")
    _emit(args, res, human)
    if skipped or not ran:
        return EXIT_SKIP
    return EXIT_PASS if passed else EXIT_FAIL


def cmd_complexity(args) -> int:
    """Decomposition checker: score a block's golden slice (or every block in
    the block diagram) on the modeling-complexity axes -- the SAME deterministic
    check the architecture Complexity Review gate runs. A block over the
    LOC/distinct-algorithm/cyclomatic budget fuses too many golden algorithms to
    be reproduced byte-exactly and should be split. Exposed so the Block Diagram
    author (and a human) can score a candidate decomposition BEFORE committing.

    Exit: PASS when all scored blocks are within budget, FAIL when any is over,
    SKIP when no golden or no block carries a python_source slice to score.
    """
    try:
        root = _bootstrap(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    try:
        from orchestrator.langgraph import block_complexity as _bc
    except Exception as exc:  # noqa: BLE001
        print(f"complexity checker unavailable: {exc}", file=sys.stderr)
        return EXIT_INFRA

    # golden path
    golden = ""
    try:
        from orchestrator.langgraph.microarch_rd import resolve_golden_path
        golden = resolve_golden_path(str(root)) or ""
    except Exception:  # noqa: BLE001
        golden = ""
    if not golden:
        _emit(args, {"skipped": True, "reason": "no golden resolvable"},
              "complexity: SKIP (no golden reference resolvable)")
        return EXIT_SKIP

    stats = _bc._parse_functions(_bc._read_golden_source(golden))

    # blocks from the live block diagram (each carries its python_source slice)
    import json as _json
    try:
        bd = _json.loads((root / ".coresmith" / "block_diagram.json")
                         .read_text(encoding="utf-8"))
        blocks = bd.get("blocks", []) or []
    except Exception:  # noqa: BLE001
        blocks = []
    if args.block:
        blocks = [b for b in blocks if b.get("name") == args.block]
        if not blocks:
            print(f"block '{args.block}' not in block_diagram.json",
                  file=sys.stderr)
            return EXIT_USAGE

    results, over = [], []
    for b in blocks:
        name = b.get("name", "")
        sl = _bc.python_source_slice_fns(b.get("python_source", ""), stats) or None
        if sl is None:
            continue  # no scoreable slice (pure memory/IO/wrapper)
        est = _bc.estimate_block_complexity(name, golden, stats=stats,
                                            slice_fns=sl)
        results.append(est)
        if est.get("over_budget"):
            over.append(est)

    if not results:
        _emit(args, {"skipped": True, "reason": "no block has a python_source slice"},
              "complexity: SKIP (no block carries a python_source slice to score)")
        return EXIT_SKIP

    human = []
    for est in results:
        tag = "OVER" if est.get("over_budget") else "ok"
        human.append(
            f"[{tag}] {est.get('block_name')}: "
            f"modeling_complexity={est.get('modeling_complexity')} "
            f"cyclo={est.get('cyclomatic')}"
            + ("".join(f"\n    - {x}" for x in est.get("axis_breaches", []))
               if est.get("over_budget") else ""))
    _emit(args, {"blocks": results, "over_budget": len(over)}, "\n".join(human))
    return EXIT_FAIL if over else EXIT_PASS


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _run(handler):
    """Wrap an int-returning handler so the CLI exits with its code."""
    def _f(args):
        raise SystemExit(handler(args))
    return _f


def _add_project_root(p) -> None:
    p.add_argument("--project-root", help="overrides $CORESMITH_PROJECT_ROOT")


def _add_json(p) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output")


def register_subcommands(sub) -> None:
    """Register harness subcommands on the ``bin/coresmith`` subparser action."""
    _register_verify(sub)
    _register_queries(sub)


def _register_queries(sub) -> None:
    # dv-status [block]
    ds = sub.add_parser("dv-status", help="DV scoreboard (per-block pass/fail)")
    _add_project_root(ds)
    _add_json(ds)
    ds.add_argument("block", nargs="?", help="restrict to one block")
    ds.set_defaults(func=_run(cmd_dv_status))

    # complexity [block] -- decomposition checker
    cx = sub.add_parser(
        "complexity",
        help="score a block's golden slice for decomposition (over-budget = "
             "fuses too many algorithms; split it)")
    _add_project_root(cx)
    _add_json(cx)
    cx.add_argument("block", nargs="?", help="restrict to one block")
    cx.set_defaults(func=_run(cmd_complexity))

    # ppa <block> [--history]
    pp = sub.add_parser("ppa", help="PPA (FF/area/cells) for a block")
    _add_project_root(pp)
    _add_json(pp)
    pp.add_argument("block")
    pp.add_argument("--history", action="store_true", help="all rows, not just latest")
    pp.set_defaults(func=_run(cmd_ppa))

    # coverage <block> [--uncovered]
    cv = sub.add_parser("coverage", help="coverage summary for a block")
    _add_project_root(cv)
    _add_json(cv)
    cv.add_argument("block")
    cv.add_argument("--uncovered", action="store_true", help="list uncovered points")
    cv.set_defaults(func=_run(cmd_coverage))

    # contracts <block>
    ct = sub.add_parser("contracts", help="interface contracts for a block")
    _add_project_root(ct)
    _add_json(ct)
    ct.add_argument("block")
    ct.set_defaults(func=_run(cmd_contracts))

    # golden-check <block>
    gc = sub.add_parser(
        "golden-check",
        help="probe a block's golden/model for degeneracy (practice tool)")
    _add_project_root(gc)
    _add_json(gc)
    gc.add_argument("block")
    gc.set_defaults(func=_run(cmd_golden_check))


def _register_verify(sub) -> None:
    """Verify subcommands (implemented in the harness ``verify`` module)."""
    # Deferred: verify handlers live in orchestrator.harness.cli_verify to keep
    # this module langgraph-free at import. Wired in the harness+verify commit.
    try:
        from orchestrator.harness import cli_verify
    except Exception:  # noqa: BLE001
        return
    cli_verify.register_verify(sub, _run, _add_project_root, _add_json)
