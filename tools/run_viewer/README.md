# CoreSmith run viewer

A read-only exporter that turns a study snapshot into a static, lazily loaded
HTML inspector of what actually ran. It covers every CLI call, every module
build and graph thread with its ordered node events and checkpoints, every
Architect, sub-agent and engine-helper trajectory, the parks and their
resolutions, and the DV / coverage / PPA rows. A label on every relationship
says how it is known.

Python stdlib only for the exporter. The UI is three dependency-free files
(`ui/index.html`, `ui/app.js`, `ui/app.css`).

## Run

```sh
# from the repository root
python3 -m tools.run_viewer.collect_study --study STUDY --output SNAP [--arm NAME ...]   # read-only capture (online SQLite backups)
python3 -m tools.run_viewer.export --snapshot SNAP --out VIEWER [--arm NAME ...] [--analysis ANALYSIS.md]
python3 -m tools.run_viewer.verify --snapshot SNAP --viewer VIEWER     # independent recount, exit 1 on mismatch
python3 -m pytest -q tools/run_viewer/tests                             # synthetic fixtures only
```

Open `VIEWER/index.html` directly (`file://` works: data shards are loaded
with `<script>` tags), or serve it:

```sh
python3 -m http.server --bind 127.0.0.1 8765 --directory VIEWER   # then http://127.0.0.1:8765/
```

The export exits with code 3 when the privacy check fails (see below).
`VIEWER/data/` is replaced on every run. The snapshot is never written:
SQLite files are opened `immutable=1`, so no `-wal` / `-shm` sidecars appear.

## Snapshot layout

```
SNAP/manifest.json           optional: collector file list (path, sha256, mode, changed_during_copy, ...)
SNAP/READY.json              optional: written by the collector when collection completed
SNAP/arms/<arm>/config.json, status.json, interventions.jsonl
SNAP/arms/<arm>/invocations/<n>/{command.json,status.json,result.json,prompt.txt,response.txt,stderr.log,transcript.jsonl}
SNAP/arms/<arm>/native-sessions/.claude/projects/**.jsonl  or  .codex/sessions/**/rollout-*.jsonl
SNAP/arms/<arm>/work/.coresmith/{project.sqlite,*_checkpoint.db,pipeline_events*.jsonl,llm_calls.jsonl,
                                 codex_turns.jsonl,live_streams/*.json,step_logs/**}
```

Nothing in the code names a particular study, arm or path.

## Views

- **Overview / arm.** State, stages, build counts, cost as reported (cumulative counters are never summed; unknown is shown as unknown), evidence issues, and a timeline of stages, builds, helpers, graph threads, parks and agents.
- **CLI ledger.** Every `actions` row (the end time of the call), interleaved with coresmith invocations seen in native shell commands that have no audit row (help, parser failures, unaudited verbs, loops). Each row has its join label and links to the native call, build, graph thread and park.
- **Builds.** One page per build:
  - the engine's own lineage predicate re-evaluated twice (active event file + exact namespace, and all event files + checkpointed ancestor namespace);
  - node iterations from the graph events, linked to helper calls;
  - checkpoints with the nodes that ran (the `versions_seen` diff) and pending-write channels;
  - CLI calls, parks, evidence rows, step logs, recorded inputs and result.
- **Graph runs.** Pipeline, backend and architecture threads: root-namespace steps, namespaces, events, builds, parks and helpers.
- **Graph events.** Every line of every `pipeline_events*.jsonl`, rotated files included, with copies referenced.
- **Trajectories.** A paged, searchable turn list per agent; the sub-agent tree; tool use ↔ result links; shell calls linked to audit rows; usage as reported. Every long field shows a preview with an explicit "Show all N characters" control.
- **Helper calls, Parks & state, Provenance** (sources, hashes, per-file counts and exclusions, labels, deduplication, privacy), **Search** (index entities and graph events, plus an on-demand search over every turn), and **Analysis** (the optional markdown file).

## How records are joined

| Label | Meaning |
| --- | --- |
| exact | an equal stable identifier: `build_id`, thread id, interrupt id, provider session/thread id, Claude `uuid`, a build id inside an audited argv |
| strong | a unique candidate on content and time: argv tokens + the audit time inside a shell call window; a module named in a helper run name + the call inside that module's only build window; a block event of the same daemon process between a build's Init Block and Block Done; `call_index` within one daemon epoch |
| weak | time only, or the closest of several compatible candidates (alternatives listed) |
| ambiguous | several equally plausible candidates, all listed, none chosen |
| unlinked | no candidate; context (scripts running at the time) is listed but not linked |

Invocation candidates are parsed only from the input of a shell tool call;
text quoted in output or documents is never parsed. A candidate in a shell
conditional or loop does not prove that the branch ran or how often it ran.
Audit rows confirm completed, audited calls. Command-position parsing
recognises the following:
- shell wrappers (`timeout`, `env`, `sudo`, ...);
- literal `for` loops, expanded with bash word-splitting;
- wrapper scripts the agent wrote (`exec coresmith "$@"`, followed through `mv` / `cp`);
- variables assigned to coresmith in the same command (`CS="${CORESMITH_CLI:-coresmith}"; "$CS" ...`).

`call_index` restarts with every daemon process, so a helper is identified by
daemon epoch and call_index together (epochs come from the writer pid of graph
events).

## Deduplication

- Claude records fold by `uuid` across the native session file, the invocation stream-json transcript and engine live-stream captures.
- Forwarded sub-agent records are routed to the sub-agent. A spawn is identified from the parent's Agent tool result or the stream's `task_started`.
- Codex response items fold by item id. `exec --json` copies (invocation transcript, `codex_turns.jsonl`, live stream) fold onto rollout items by output or text digest, in order. Exec item ids restart per call and are never compared across calls.
- Graph events fold only when the identical whole record appears in another file; the k-th occurrence pairs with the k-th earlier copy. Repeats inside one file are kept.

## Privacy

Content is exported by allowlist. The following become length markers:
- hidden reasoning: Claude `thinking` / `redacted_thinking`, streamed `thinking_delta`, Codex `reasoning`, `raw_content`, `summary_text`, analysis-channel messages;
- signatures and encrypted payloads;
- private fields that appear inside visible text: an agent that printed a raw session record, at any JSON escaping depth, including values cut off mid-string.

Credential attachments are dropped and credential-like tokens redacted.
`decisions.reasoning` and resolution rationales in the engine database are
operator-supplied public text and are kept.

Every withheld payload leaves fingerprints. Signatures and encrypted payloads
are fingerprinted at a 40-character stride, so a copied fragment of 79+
characters is caught and withheld wherever it appears. The final export is
scanned for every fingerprint and for encrypted-payload patterns; any hit
fails the export. A narration thinking block whose whole text is also the
same agent's visible output is not treated as private. A partial overlap
never exempts anything.

## Limits

- A snapshot taken from a live run is not atomic across files. Records written after a file was copied are absent, and in-flight builds and calls show their state at copy time.
- "Published at snapshot" is `results.best`. Freshness against live inputs cannot be re-evaluated without the RTL and build artifacts.
- CLI ↔ native joins are heuristic by construction: the audit stores no invocation id or start time. Helper ↔ build joins are strong, because `llm_calls.jsonl` stores no build or thread id.
- Visible provider records are preserved as captured. Output already truncated by the provider cannot be reconstructed.
