# SystemC TLM-2.0 loosely-timed block models

Rules of thumb for a model that is fast, wireable and comparable to RTL:
- One `SC_THREAD(run)` per block; `wait()` on the clock only where the spec
  spends cycles. Never spin without `wait()`.
- Targets: implement `b_transport_<member>`; decode `trans.get_address()`,
  honour `TLM_READ_COMMAND` / `TLM_WRITE_COMMAND`, copy exactly
  `get_data_length()` bytes, set the response status, add the contract latency
  to `delay`. Use `cs_mem` for byte-addressed state.
- Initiators: build the request with `cs_transact(...)`; pass the returned
  status up as the spec's fault/error field.
- Streams: `fifo.write(beat)` / `fifo.read()`; pack the contract's fields
  LSB-first into the 64-bit beat; a beat wider than 64 bits is sent as
  consecutive beats, LSB word first.
- Static wires: `sc_out.write(v)` once per change; consumers `read()`.
- `dump_state` prints `name=value` lines for every architectural register and
  a short summary of memories (size, checksum) -- it is diffed against RTL.
- Deterministic: no random numbers, no wall-clock time, no environment reads.
- Completeness beats speed: the FRD is evaluated on the assembled model
  before RTL. Implement the whole register map and every operation the uArch
  spec lists; a processor block executes its ISA subset functionally; memories
  hold real contents (`cs_mem`, preloadable with `load_bin`).
- Expose architectural state publicly (members/getters) for the FRD harness;
  keep `dump_state` complete -- it is diffed against RTL later.
- The fabric router has an extra optional master port `s_cs_tester`; the FRD
  harness's `cs_tester` binds there to reach every memory-mapped slave.
