# Run supervision

The Architect -- the coding agent that talks to the human and drives the
`coresmith` CLI -- is outside the engine. The daemon never launches, resumes
or decides for it: parks wait for the Architect's own `coresmith resume`, and
`coresmith stage next` / `coresmith run start` are the Architect's calls.

`CORESMITH_SUPERVISOR=1` (opt-in, set before `coresmith daemon start`) adds one
deterministic responsibility: once the stage machine has reached `blocks` and
no run exists yet, start the frontend pipeline once (a durable, budgeted
handoff recorded in the project database). Before `blocks` it reports
`running` with `stage <s>: waiting for the Architect to reach blocks` and
launches nothing.

```
coresmith supervisor check
coresmith supervisor status
coresmith supervisor retry
```

Cron can run `coresmith supervisor check` every 30 minutes after sourcing the
run environment. Its only recovery responsibility is restarting a dead daemon
(maximum three starts/hour). Pipeline checkpoints are never reset and parked
decisions are never answered by supervision.

A cluster worker's (`CORESMITH_FANOUT=cluster`) provider or tool failure is
reported as infrastructure state (`blocked`, `failure.worker`), never as a
design verdict; the reason is the runner's own diagnostic (its stderr tail),
never the model's last message. After repairing the cause, `coresmith
supervisor retry` clears the infrastructure block; simply restarting the
daemon does not discard it. A stale `.coresmith/architect/status.json` from an
older engine's Architect loop is ignored.

Status is available through the CLI, `GET /supervisor/status`, the project DB,
and `.coresmith/supervisor/status.json`. Transitions are appended to
`.coresmith/supervisor/events.jsonl`; blocked transitions also appear in daemon
logs. These are local records, not outbound notifications.

`state` additionally reports `passed_count`, `failed_count`, and `pending_count`.
The legacy `completed_count` retains its meaning of attempted terminal blocks
for compatibility; it must not be presented as passed blocks.

Frontend completion comes from the daemon state, with pending decisions and
failed blocks checked first. `frontend_complete` is not whole-chip acceptance.
The daemon refuses startup without its project lease, and stops its owned work
if it cannot retain ownership.
