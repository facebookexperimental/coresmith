SIM TIMEOUT ENGINE FIX -- 2026-06-23
====================================

Problem (OTEL deep-dive, run codecv4-synthgated-20260622-2140)
------------------------------------------------------------
The per-block cocotb/Verilator SIM in run_simulation() (pipeline_helpers.py)
had a bare hardcoded cap. intra_rd_encode_core's full-datapath RTL is slow on
larger seeded scenarios and exceeded it; the timeout message "...timed out..."
matched _INFRA_MARKERS in diagnose_node, so 4 of 5 attempts were classified
INFRASTRUCTURE_ERROR and the block escalated WITHOUT codex ever seeing a
functional verdict. Only attempt 5 finished and produced a real PSNR
divergence. The diagnose->revise loop was STARVED by timeouts.

Fix (files, both /coresmith and /app copies)
-------------------------------------------
1. orchestrator/langgraph/pipeline_helpers.py  run_simulation():
   - Sim timeout resolved as MAX of three channels, then logged:
       [SIM] timeout=Ns (default|declared|extended; env=.. declared=.. ..)
     (a) env default: CORESMITH_SIM_TIMEOUT_S (default 900s / 15 min),
         scaled() so CORESMITH_TIMEOUT_MULTIPLIER still applies.
     (b) per-block declared runtime target, read from the block dict:
         sim_timeout_s | runtime_target_s | requested_sim_timeout_s
         (the uArch/agent or diagnose structured output can set any of these).
     (c) auto-extend: x1.5 per prior TIMEOUT for THIS block (persisted in
         sim_build/<block>/sim_timeout_state.json so it survives daemon
         restarts), capped at CORESMITH_SIM_TIMEOUT_CAP_S (default 1800s).
   - On TimeoutExpired: increments the per-block timeout counter, and returns
     a distinct message "SIM_TIMEOUT: Simulation exceeded Ns (M min)..." plus
     sim_timed_out=True (NOT the generic "timed out" infra marker).

2. orchestrator/langgraph/pipeline_graph.py  diagnose_node():
   - New short-circuit (before the INFRA short-circuit): a "SIM_TIMEOUT:" /
     "Simulation exceeded" error is classified as its OWN category SIM_TIMEOUT
     (confidence 0, no LLM call) -- a pure timeout has no verdict to diagnose.

3. orchestrator/langgraph/pipeline_graph.py  _route_decision():
   - New Rule -1: SIM_TIMEOUT -> retry_rtl (re-runs sim with the auto-extended
     cap) until CORESMITH_SIM_TIMEOUT_MAX_RETRIES (default 4) timeouts, then
     escalate. Does NOT trip the INFRASTRUCTURE_ERROR(>=2 -> ask_human) or
     same-category(>=3 -> escalate) budgets.

4. orchestrator/langgraph/pipeline_graph.py  decide_node():
   - A SIM_TIMEOUT retry does NOT increment the functional attempt counter
     (state["attempt"]); it re-runs the same attempt# with more wall-clock.
     A pure timeout no longer consumes the max_attempts diagnose budget, so a
     slow-but-correct block gets real diagnoses instead of wasted rounds.

New env knobs
-------------
CORESMITH_SIM_TIMEOUT_S=900            base/default per-block sim cap (15 min)
CORESMITH_SIM_TIMEOUT_CAP_S=1800       hard cap for auto-extension
CORESMITH_SIM_TIMEOUT_MAX_RETRIES=4    max pure-timeout retries before escalate
(plus existing CORESMITH_TIMEOUT_MULTIPLIER still multiplies all of the above)

Validation (real Verilator sims, no gaming)
-------------------------------------------
- py_compile clean on all 4 files.
- Resolution arithmetic unit-tested: default 900 / env override / declared-max
  / extend x1.5 capped 1800 / multiplier-aware -- all correct.
- run_simulation() invoked on the REAL on-disk intra_rd_encode_core RTL+TB:
  COMPLETED in 6.2s, [SIM] timeout=900s logged, returned a REAL functional
  verdict (dut_psnr=10.67 dB vs model 49.25 dB) -- codex now has a PSNR
  divergence to root-cause, not a timeout.
- Declared (1500s) and auto-extend (x1.5^2 -> 1800 cap) channels confirmed
  live through run_simulation's [SIM] log.

Honest residual
---------------
The timeout STARVATION is fixed. The encoder RTL itself is still at the
full-RD-rewrite wall: on-disk RTL reconstructs at ~10-17 dB vs the ~42-49 dB
block model (genuine LOGIC_ERROR, conf 0.95). That is a real design bug for
the diagnose->revise loop (now fed) to fix; it is not an infrastructure issue.
