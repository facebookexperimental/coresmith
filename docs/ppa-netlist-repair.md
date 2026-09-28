# Pre-placement netlist repair

The PPA timing check asks the active deployment for the optional
`repair_netlist` tool after ABC mapping. Sky130 implements it with OpenROAD's
`repair_design -pre_placement`, using deployment-owned LEFs, site, fanout limit,
cell exclusions and tool resolution. The Liberty and clock period are those of
the timing measurement, so repair does not silently switch corners.

ABC's extracted combinational network does not represent every mapped flip-flop
load. In a DVB-T run, a nominally buffered netlist still had one net driving
1,536 flip-flop D pins. Standard repair closed timing without changing RTL;
asking the RTL worker to restructure that memory was avoidable.

After repair, a fresh OpenSTA process reads the written netlist. Yosys measures
area, cells and flip-flops on that same artifact. The pipeline selects a measured
candidate as a unit: its timing, area, FF count, report and netlist hash travel
together. Added buffers count toward any declared area budget. Reports and the
repair script/log are retained beside the selected netlist when a report
directory is supplied.

Missing repair capability or failed repair is explicit in `repair_status` and
`detail`. The existing measured base/ABC fallback remains available, with its
own area and timing; failure is never reported as a successful repair. Old
outputs, absent completion, nonzero exits and missing PDK files cannot establish
a successful repair. No parseable timing or mapped area means that candidate
is unavailable, not a passing measurement.

This is a pre-placement estimate with ideal clocks and no placed/routed wire
parasitics. It is not routed timing signoff. Block DV, integration DV and final
validation are unchanged, for both single-block and multi-block designs.

The tool can also be invoked through the usual CLI:

```sh
"${CORESMITH_CLI:-coresmith}" tool repair_netlist --design core \
  --netlist mapped.v --liberty corner.lib --clock-ns 15.625 \
  --clock-port clk --out-dir repair --timeout-s 300 --json
```

Pure regressions are in `orchestrator/tests/test_ppa_netlist_repair.py`. Its
optional real-tool test creates a 1,536-entry flop store whose unbuffered timing
fails, verifies repaired timing passes, and counts the mapped loads to confirm
the deployment's fanout limit. Run it with `CORESMITH_TEST_PDK_ROOT` set to a
Sky130 installation, `yosys` and `sta` on PATH, and OpenROAD resolved through
the deployment (or `CORESMITH_BACKEND_OPENROAD`).
