# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test-baseline guard: make a NEW test failure visible against a red suite.

``orchestrator/tests`` has been red for a long time with failures that predate
current work.  With ~100 pre-existing red tests nobody can tell whether their
change added the next one, so every session re-litigates "is this failure
mine?".  This script answers that mechanically:

    orchestrator/tests/baseline_failures.txt   the ledger of ALREADY-red tests
    scripts/check_test_baseline.py             runs the suite, diffs it

Exit codes (same convention as ``bin/coresmith``: 0 ok, 1 verdict-fail,
2 could-not-judge):

    0   no test is failing that is not already in the baseline
    1   REGRESSION -- at least one node ID fails that the baseline does not list
    2   the guard itself could not run (pytest crashed, baseline missing, ...)

Tests in the baseline that now PASS are reported but never fail the guard --
they are progress, and the message tells you to prune them.  Baseline entries
that no longer exist (renamed, deleted, deselected by a marker filter) are
reported as stale, not crashed on.

Usage::

    # the gate: did I add the next failure?
    python scripts/check_test_baseline.py

    # deliberate refresh after paying debt down (or after a rename)
    python scripts/check_test_baseline.py --refresh

    # judge a dump you already have, without paying for another ~200s run
    python scripts/check_test_baseline.py --from-json /tmp/run.json

Why a script and not a pytest test: the guard has to run the WHOLE suite as a
subprocess.  A pytest test that does that would collect itself inside its own
child run and re-spawn the suite recursively.  It is also a CI/pre-push gate
whose product is an exit code, which is exactly what ``scripts/nightly_canary.py``
and ``scripts/triage_escalation.py`` already are.

See ``docs/TEST-BASELINE.md`` for the refresh policy.  The baseline is a debt
ledger to shrink, not a permanent excuse.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "orchestrator" / "tests" / "baseline_failures.txt"

# The invocation the baseline is measured with. Deliberately NOT marker-filtered:
# the baseline records the whole suite as it actually behaves on this machine, so
# a run that filters markers is a DIFFERENT population and is flagged as such.
PYTEST_ARGS = [
    "-m",
    "pytest",
    "orchestrator/tests",
    "-q",
    "--no-header",
    "-p",
    "no:cacheprovider",
]

# Written to a temp dir and loaded with `-p`. Parsing pytest's `-rf` short
# summary instead would be ambiguous: the summary line is "FAILED <nodeid> - <msg>"
# and a parametrized node ID may itself contain " - ".
_COLLECTOR_PLUGIN = '''
"""Dump this run's per-test outcomes as JSON. Written by check_test_baseline.py."""
import json
import os

_outcomes = {}
_collected = set()


def pytest_collection_modifyitems(items):
    for item in items:
        _collected.add(item.nodeid)


def pytest_collectreport(report):
    # A module that fails to import yields a collect error whose nodeid is the
    # file path, and never produces items -- count it as both seen and failed.
    if report.failed:
        _collected.add(report.nodeid)
        _outcomes[report.nodeid] = "failed"


def pytest_runtest_logreport(report):
    nodeid = report.nodeid
    _collected.add(nodeid)
    if report.failed:
        _outcomes[nodeid] = "failed"
    elif report.skipped:
        if _outcomes.get(nodeid) != "failed":
            _outcomes[nodeid] = "xfailed" if hasattr(report, "wasxfail") else "skipped"
    elif report.when == "call" and nodeid not in _outcomes:
        _outcomes[nodeid] = "xpassed" if hasattr(report, "wasxfail") else "passed"


def pytest_sessionfinish(session, exitstatus):
    counts = {}
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        for key, reports in reporter.stats.items():
            if key:
                counts[key] = len(reports)
    payload = {
        "outcomes": _outcomes,
        "collected": sorted(_collected),
        "counts": counts,
        "exitstatus": int(exitstatus),
    }
    with open(os.environ["CS_BASELINE_DUMP"], "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
'''

_HEADER_PREAMBLE = """\
# CoreSmith test-suite failure baseline.
#
# THIS IS A DEBT LEDGER, NOT A PERMANENT EXCUSE. Every node ID below is a test
# that was ALREADY failing before the current work started. The list exists for
# exactly one reason: so that failure number N+1 -- the one YOU just added -- is
# visible against a suite that is already red.
#
# Rules:
#   * Never add a line to this file to make your own change go green. Adding a
#     line is admitting a regression. Fix the test or fix the code.
#   * Deleting lines is the point. When a test here starts passing, prune it:
#       python scripts/check_test_baseline.py --refresh
#   * The guard reads only the node IDs. Header fields are provenance.
#
# The failing set depends on the TOOLCHAIN, not only the code -- see the
# `environment:` field below. On the first measurement 50 of 75 entries failed
# only because the `claude` binary was off PATH. Put the toolchain back before
# you believe a big diff.
#
# Check a change against it:   python scripts/check_test_baseline.py
# Full policy:                 docs/TEST-BASELINE.md
#
"""


# --------------------------------------------------------------------------- #
# baseline file I/O
# --------------------------------------------------------------------------- #


def parse_baseline(path: Path) -> tuple[set[str], dict[str, str]]:
    """Return ``(node_ids, header_fields)`` from a baseline file.

    Lines starting with ``#`` are comments; ``# key: value`` comments are also
    harvested as provenance fields. Blank lines are ignored.
    """
    node_ids: set[str] = set()
    header: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            match = re.match(r"^#\s*([a-z][a-z0-9-]*)\s*:\s*(.*)$", line)
            if match:
                header[match.group(1)] = match.group(2).strip()
            continue
        node_ids.add(line)
    return node_ids, header


def probe_environment() -> str:
    """Record the environment facts that move whole BLOCKS of this suite.

    Two thirds of the failures in the first baseline were not code at all: 50
    tests failed with "Claude CLI not found" because ``claude`` was not on PATH,
    and the PDK tests failed because ``.pdk/`` was absent. A ledger that does not
    say which tools were present invites a future reader to mistake a PATH change
    for fifty regressions (or fifty repairs).
    """
    pdk_root = os.environ.get("PDK_ROOT") or str(REPO_ROOT / ".pdk")
    facts = [
        ("claude", bool(shutil.which("claude"))),
        ("pdk", Path(pdk_root).is_dir()),
    ]
    facts += [(tool, bool(shutil.which(tool)))
              for tool in ("yosys", "verilator", "openroad", "magic", "iverilog")]
    return " ".join(f"{name}={'yes' if ok else 'NO'}" for name, ok in facts)


def render_baseline(failures: set[str], counts: dict[str, int], commit: str,
                    invocation: str) -> str:
    """Render a baseline file body (header + sorted, de-duplicated node IDs)."""
    ordered = ("failed", "passed", "skipped", "xfailed", "xpassed", "error")
    shown = [f"{key}={counts[key]}" for key in ordered if counts.get(key)]
    extra = [f"{k}={v}" for k, v in sorted(counts.items())
             if k not in ordered and isinstance(v, int)]
    lines = [_HEADER_PREAMBLE.rstrip("\n")]
    lines.append(f"# measured-at-commit: {commit}")
    lines.append(f"# measured-on: {date.today().isoformat()}")
    lines.append(f"# pytest-invocation: {invocation}")
    lines.append(f"# environment: {probe_environment()}")
    lines.append(f"# counts: {' '.join(shown + extra)}")
    lines.append(f"# baseline-failures: {len(failures)}")
    lines.append("#")
    lines.extend(sorted(failures))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# running the suite
# --------------------------------------------------------------------------- #


def describe_invocation(extra_args: list[str]) -> str:
    return " ".join(["python", *PYTEST_ARGS, *extra_args])


def run_suite(extra_args: list[str], keep_json: Path | None) -> dict:
    """Run the full suite and return the collector plugin's JSON payload."""
    with tempfile.TemporaryDirectory(prefix="cs-baseline-") as tmp:
        plugin_dir = Path(tmp)
        (plugin_dir / "_cs_baseline_collector.py").write_text(_COLLECTOR_PLUGIN)
        dump_path = plugin_dir / "dump.json"

        env = dict(os.environ)
        env["CS_BASELINE_DUMP"] = str(dump_path)
        pythonpath = [str(REPO_ROOT), str(plugin_dir)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        argv = [sys.executable, *PYTEST_ARGS, "-p", "_cs_baseline_collector",
                "--tb=no", *extra_args]
        print(f"[baseline] running: {' '.join(argv)}", flush=True)
        proc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env)

        if not dump_path.exists():
            raise RuntimeError(
                f"pytest exited {proc.returncode} without writing a run dump -- "
                "the suite probably crashed (segfault / collection abort / OOM) "
                "rather than merely failing. Re-run pytest by hand to see why."
            )
        payload = json.loads(dump_path.read_text())
        if keep_json:
            keep_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"[baseline] raw run dump kept at {keep_json}")
        return payload


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --------------------------------------------------------------------------- #
# the diff
# --------------------------------------------------------------------------- #


def judge(payload: dict, baseline: set[str], header: dict[str, str],
          invocation: str) -> int:
    outcomes: dict[str, str] = payload.get("outcomes", {})
    collected = set(payload.get("collected", []))
    counts: dict[str, int] = payload.get("counts", {})
    now_failed = {nodeid for nodeid, outcome in outcomes.items() if outcome == "failed"}

    new_failures = sorted(now_failed - baseline)
    seen = baseline & collected
    now_passing = sorted(n for n in seen if outcomes.get(n) in ("passed", "xpassed"))
    now_skipped = sorted(n for n in seen if outcomes.get(n) in ("skipped", "xfailed"))
    # Renamed / deleted / deselected: in the ledger but the run never saw it.
    vanished = sorted(baseline - collected)

    shown = " ".join(f"{k}={counts[k]}" for k in
                     ("failed", "passed", "skipped", "xfailed", "xpassed", "error")
                     if counts.get(k))
    print()
    print("=" * 72)
    print(f"this run:  {shown or 'no counts reported'}")
    print(f"baseline:  {len(baseline)} known-red node IDs "
          f"(measured at {header.get('measured-at-commit', '?')[:12]} "
          f"on {header.get('measured-on', '?')})")

    recorded = header.get("pytest-invocation")
    if recorded and recorded != invocation:
        print()
        print("WARNING: this run's invocation differs from the one the baseline "
              "was measured with.")
        print(f"  baseline: {recorded}")
        print(f"  this run: {invocation}")
        print("  A marker-filtered or subset run is a different population; the "
              "diff below may be noise.")

    recorded_env = header.get("environment")
    current_env = probe_environment()
    if recorded_env and recorded_env != current_env:
        print()
        print("WARNING: the toolchain present now differs from when the baseline "
              "was measured.")
        print(f"  baseline: {recorded_env}")
        print(f"  this run: {current_env}")
        print("  Whole blocks of this suite are gated on these tools (~50 tests need "
              "the `claude`")
        print("  binary alone). A large swing below is probably PATH, not code.")

    if now_passing:
        print()
        print(f"PROGRESS -- {len(now_passing)} baseline failure(s) now PASS. "
              "Prune them from the ledger:")
        for nodeid in now_passing:
            print(f"  + {nodeid}")
        print("  ->  python scripts/check_test_baseline.py --refresh")

    if now_skipped:
        print()
        print(f"NOTE -- {len(now_skipped)} baseline failure(s) are now skipped/xfailed, "
              "not fixed:")
        for nodeid in now_skipped:
            print(f"  ~ {nodeid} ({outcomes.get(nodeid)})")

    if vanished:
        print()
        print(f"STALE -- {len(vanished)} baseline entry/entries were not collected at all "
              "(renamed, deleted, or deselected):")
        for nodeid in vanished:
            print(f"  ? {nodeid}")
        print("  ->  refresh the baseline if these tests are genuinely gone.")

    if new_failures:
        print()
        print(f"FAIL -- {len(new_failures)} NEW test failure(s) not in the baseline:")
        for nodeid in new_failures:
            print(f"  ! {nodeid}")
        print()
        print("These are yours. Reproduce one with:")
        print(f"  python -m pytest '{new_failures[0]}' -x -vv")
        print("If it fails only inside the full suite it is an order-dependence bug, "
              "still yours.")
        print("=" * 72)
        return 1

    print()
    print(f"OK -- no failure outside the baseline ({len(now_failed)} failing, "
          f"all {len(now_failed & baseline)} already known-red).")
    print("=" * 72)
    return 0


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="baseline ledger to read/write (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true",
                        help="DELIBERATE re-measure: rewrite the baseline from this run "
                             "instead of judging against it")
    parser.add_argument("--from-json", type=Path, metavar="PATH",
                        help="reuse a dump written by an earlier --json-out run instead "
                             "of re-running the suite")
    parser.add_argument("--json-out", type=Path, metavar="PATH",
                        help="keep this run's raw dump at PATH")
    parser.add_argument("pytest_args", nargs="*",
                        help="extra args appended to the pytest invocation (changes the "
                             "population; the guard warns when it does)")
    args = parser.parse_args(argv)

    invocation = describe_invocation(args.pytest_args)

    if args.from_json:
        if not args.from_json.exists():
            print(f"ERROR: no such dump: {args.from_json}", file=sys.stderr)
            return 2
        try:
            payload = json.loads(args.from_json.read_text())
        except (OSError, ValueError) as exc:
            print(f"ERROR: unreadable dump {args.from_json}: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            payload = run_suite(args.pytest_args, args.json_out)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    outcomes: dict[str, str] = payload.get("outcomes", {})
    counts: dict[str, int] = payload.get("counts", {})
    failed = {nodeid for nodeid, outcome in outcomes.items() if outcome == "failed"}

    if not outcomes:
        print("ERROR: the run dump records no test outcomes at all -- collection "
              "failed before any test ran.", file=sys.stderr)
        return 2

    if args.refresh:
        body = render_baseline(failed, counts, git_commit(), invocation)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(body)
        print(f"\n[baseline] wrote {len(failed)} known-red node IDs to {args.baseline}")
        print("[baseline] REVIEW THE DIFF before committing. Lines ADDED are "
              "regressions you are about to bless;")
        print("[baseline] lines REMOVED are debt you paid off.")
        return 0

    if not args.baseline.exists():
        print(f"ERROR: no baseline at {args.baseline}. Create one deliberately, from a "
              "PRISTINE tree:\n"
              "  python scripts/check_test_baseline.py --refresh", file=sys.stderr)
        return 2
    try:
        baseline, header = parse_baseline(args.baseline)
    except OSError as exc:
        print(f"ERROR: unreadable baseline {args.baseline}: {exc}", file=sys.stderr)
        return 2
    if not baseline:
        print(f"ERROR: baseline {args.baseline} lists no node IDs.", file=sys.stderr)
        return 2

    return judge(payload, baseline, header, invocation)


if __name__ == "__main__":
    sys.exit(main())
