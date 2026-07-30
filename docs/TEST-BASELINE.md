<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
-->

# The test baseline (debt ledger)

`pytest orchestrator/tests` is **red on `main`** and has been for a long time.
Roughly a hundred tests fail for reasons that predate whatever you are working
on — assertion strings that drifted, tests pinned to APIs that were renamed,
tests that need EDA tools or a live LLM that this box does not have.

That is a problem beyond the failures themselves: **when the suite is already
red, nobody can tell whether a change added the next failure.** Every session
re-discovers the red suite, re-triages the same hundred failures, and burns time
deciding "is this one mine?".

Two files fix that:

| Path | What it is |
| --- | --- |
| `orchestrator/tests/baseline_failures.txt` | The ledger: every node ID that was **already** failing, one per line, sorted, with a header recording the commit / date / invocation / counts it was measured at. |
| `scripts/check_test_baseline.py` | The guard: runs the suite, diffs it against the ledger, exits non-zero **only** for failures the ledger does not list. |

**The ledger is a debt ledger, not a permanent excuse.** Its only correct
direction of travel is shorter. See [Paying it down](#paying-it-down).

## Use it

```bash
# from the repo root, with the venv + EDA tools on PATH
python scripts/check_test_baseline.py
```

Exit codes follow the `bin/coresmith` convention:

| Code | Meaning |
| --- | --- |
| `0` | Every failing test is already in the ledger. Your change added nothing. |
| `1` | **Regression.** At least one node ID fails that the ledger does not list — it is printed, and it is yours. |
| `2` | The guard could not judge: pytest crashed without producing a report, or the ledger is missing/empty. |

The guard is only as good as the PATH you run it with — see
[The toolchain moves the ledger](#the-toolchain-moves-the-ledger).

Beyond the pass/fail verdict, a run also reports (**without** failing):

* **PROGRESS** — ledger entries that now pass. Prune them.
* **NOTE** — ledger entries that are now `skipped`/`xfailed`. Marked as skipped
  is *not* the same as fixed; these are called out separately so a skip cannot
  quietly masquerade as a repair.
* **STALE** — ledger entries the run never collected: renamed, deleted, or
  filtered out by a marker selection. The guard reports and continues rather
  than crashing on a node ID that no longer resolves.

Useful flags:

```bash
# keep the raw per-test dump, then re-judge it for free (no second 200s run)
python scripts/check_test_baseline.py --json-out /tmp/run.json
python scripts/check_test_baseline.py --from-json /tmp/run.json

# judge against a ledger somewhere else
python scripts/check_test_baseline.py --baseline /tmp/other_baseline.txt
```

Anything after the flags is appended to the pytest invocation. That **changes
the population** being measured (a marker-filtered run collects fewer tests), so
the guard compares your invocation against the one recorded in the ledger header
and prints a warning when they differ. Treat the diff from a filtered run as
advisory only.

## The toolchain moves the ledger

**Most of the first baseline was not code.** Of the 75 entries measured at
`06d82e8`:

| Count | Cause | Cost to pay down |
| --- | --- | --- |
| 50 | `FileNotFoundError: Claude CLI not found` — tests instantiate `ClaudeLLM()` at import/construct time, and `claude` was not on PATH | Put `claude` on PATH. Zero code changes. |
| 15 | `orchestrator/tests/test_composition_audit.py` — MyHDL-style fixtures vs an auditor migrated to Amaranth `Elaboratable` | Real work; the file's own docstring already documents it |
| 6 | `orchestrator/tests/test_microarch_exp.py` — same Amaranth migration | Same |
| 1 | `test_pdk_files_exist` — no `.pdk/` on this box | Install the PDK |
| 3 | Genuine assertion drift / real red | Cheap, one line each |

That is why the ledger header carries an `environment:` field, and why the guard
warns when the toolchain present now differs from the toolchain the ledger was
measured with. Without it, someone adding `claude` to PATH sees 50 tests "get
fixed", and someone dropping it off PATH sees 50 "regressions" — neither is a
code change.

Before you trust any large diff: check the `environment:` line first.

## Refreshing the ledger, deliberately

Refreshing is a **deliberate, reviewed act**, never a way to get to green.

```bash
# 1. Measure from a PRISTINE tree, not from your feature branch. A baseline
#    taken on a branch with work in flight bakes that work's failures in.
git worktree add --detach /tmp/wt-baseline origin/main
cd /tmp/wt-baseline

# 2. Copy in the guard if the pristine commit predates it, then measure.
python scripts/check_test_baseline.py --refresh \
    --baseline /tmp/baseline_failures.txt

# 3. Read the diff line by line before committing it.
diff -u orchestrator/tests/baseline_failures.txt /tmp/baseline_failures.txt

# 4. Clean up.
cd - && git worktree remove /tmp/wt-baseline
```

When reviewing that diff:

* **Removed lines are the whole point.** A test dropped out of the ledger
  because it passes now. Good. Commit it.
* **Added lines are a confession.** Every added line says "I am blessing a
  newly-red test". The default answer is *no*: fix the test or fix the code.
  An added line needs a reason in the commit message, and it should be an
  external one (a dependency bumped, a tool disappeared from the box) — never
  "my change broke it".
* **A ledger that grew is a failed refresh.** If the count went up and you
  cannot name why, throw the refresh away.

Because the failing set depends on what is installed (EDA tools, `claude` CLI,
Nix), a refresh on a different machine can legitimately shift the list. The
header's `pytest-invocation` and `counts` fields exist so you can tell an
environment shift from a code regression before you believe the diff.

## Paying it down

The ledger is sorted by node ID, so the cheap wins cluster by file. In priority
order for the `06d82e8` measurement:

1. **Put `claude` on PATH** (`/home/ubuntu/.npm-global/bin` on the OCI box; CI
   already `npm install -g @anthropic-ai/claude-code`s it). Retires ~50 of 75
   entries with no code change at all.
2. **`test_rtl_model_equiv.py::test_non_axis_interface_skips`** — asserts
   `"AXI-Stream" in res["reason"]`, but the reason string the code emits today
   reads `"... the default AXIS harness covers single s_axis-in / m_axis-out ..."`.
   One drifted word. One line.
3. **`test_composition_audit.py` (15) + `test_microarch_exp.py` (6)** — one root
   cause: the fixtures are pre-migration MyHDL `@block` functions and the auditor
   now requires Amaranth `class X(Elaboratable)`. Migrating the fixtures retires
   21 entries in one commit. `test_composition_audit.py`'s module docstring
   already spells this out.
4. **`test_pdk_files_exist`** — wants `.pdk/sky130A/.../sky130_fd_sc_hd__nom.tlef`.
   Environmental; install the PDK or leave it in the ledger and say so.

Attack the trivial classes first — a drifted assertion string costs one line and
buys back real signal.

Workflow for paying down debt:

1. Fix the test (or the code — decide which one is actually wrong; a drifted
   assertion string is usually the test's fault, but an assertion that stopped
   matching because behavior silently changed is the *code's* fault and is a
   real bug the ledger has been hiding).
2. Re-run the guard. The fixed test shows up under **PROGRESS**.
3. `--refresh` and commit the shrunken ledger with the fix.

Do not batch a hundred fixes into one refresh. One class of failure per commit
keeps the ledger diff readable, which is the only thing that keeps it honest.

## Design notes

**Why a script, not a pytest test.** The guard has to run the whole suite as a
subprocess. As a pytest test it would be collected inside its own child run and
re-spawn the suite recursively. It is also a gate whose product is an exit code
for CI or a pre-push hook — the same shape as `scripts/nightly_canary.py` and
`scripts/triage_escalation.py`, which is where this repo already puts
"run the thing, judge the result, exit non-zero" logic.

**Why a plain-text ledger.** One node ID per line, sorted, no quoting: a new or
retired failure is exactly one line of diff in review, which makes an
inflationary refresh impossible to sneak past a reviewer.

**How failures are collected.** The script writes a small pytest plugin to a
temp dir and loads it with `-p`, so outcomes come from `pytest_runtest_logreport`
/ `pytest_collectreport` rather than from scraping terminal output. The short
summary line is `FAILED <nodeid> - <msg>` and a parametrized node ID can itself
contain `" - "`, so scraping is ambiguous; hooks are not. Module-level import
errors surface as collect-report failures whose node ID is the file path, and are
recorded too.

**This box has no `pytest-timeout`.** Passing `--timeout=` to any invocation
here is an error, not a slow run.
