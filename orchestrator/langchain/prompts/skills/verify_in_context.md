# Skill: verify in context (pull state via the `coresmith` CLI)

You have an engine-owned verification CLI. Use it to CHECK your work before you
finish a turn, and to PULL just the context you need instead of relying on a
giant inlined prompt dump.

## Invoke it via `$CORESMITH_CLI` (PATH is unreliable in your shell)
The daemon exports **`CORESMITH_CLI`** — the absolute path to the CLI. Your
commands run under `/bin/bash -lc`, where `/etc/profile` can reassign `PATH` and
drop the `coresmith` shim (a bare `coresmith` then fails with exit 127). So
ALWAYS call it through `$CORESMITH_CLI`, falling back to a plain `coresmith`
only if the variable is unset. Define a short alias once per command block:

```bash
CS="${CORESMITH_CLI:-coresmith}"
"$CS" verify rtl <block> --json
```

Every `"$CS" ...` example below uses that convention.

## Verify before finishing
Run the matching check and read its verdict (exit 0 = pass, 1 = fail,
3 = infra/timeout, 4 = skip/cannot-judge; add `--json` for machine output):

- `"$CS" verify model <block>` — elaborate + interface + size an Amaranth block
  model (seconds; the fast pre-RTL check).
- `"$CS" verify rtl <block>` — lint + cocotb sim + RTL-vs-model byte-exact
  equivalence for one block.
- `"$CS" verify synth <block>` — synthesizability / PPA (FF + cell budget).
- `"$CS" verify chip-model` / `"$CS" verify chip` — composed-model and
  integrated chip_top checks.

## Pull state instead of re-deriving it
- `"$CS" contracts <block>` — this block's interface contracts (producer /
  consumer edges + packing convention). Read the block's `bootstrap_policy`
  (esp. `reset_seed`) from here rather than guessing.
- `"$CS" dv-status [block]` — per-block DV pass/fail from the scoreboard.
- `"$CS" ppa <block>` — FF / cell / area for the block.
- `"$CS" coverage <block> --uncovered` — uncovered coverage points.

## The gate re-runs with a FRESH seed — only that run counts
Your local `"$CS" verify` runs are advisory (`source=agent`). At gate-accept
the engine re-runs the SAME function with a fresh, unpredictable seed
(`source=gate`) — that is the run of record. A testbench that only passes on a
pinned seed will FAIL the gate. Implement the real datapath; do not memorize
vectors.

## Oracle tampering is detected
The golden reference, the stimulus under `inputs/`, and the arch specs are
SHA-256-pinned at run start. Editing any of them to make your RTL "match" is
detected as an `ORACLE_TAMPER` failure. Fix the RTL, never the oracle.
