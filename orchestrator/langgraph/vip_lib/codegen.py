# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render one VIP module (cocotb) and one SVA bind file per contract edge.

Pure string templates over ``EdgeContract`` -- deterministic, so the same
contract always yields byte-identical VIP code (fingerprinted).

Sampling convention of the generated cocotb code (it is what makes the
contract's cycle counting unambiguous):

* drivers change DUT inputs just after a rising edge (``RisingEdge`` + one
  delta step);
* monitors and assertions sample at the FALLING edge, when the inputs driven
  this cycle and the registered outputs updated at this cycle's rising edge
  are all stable "current-cycle" values;
* a request accepted in cycle t and answered in cycle t+N has latency N.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .contract_model import EdgeContract

VIP_DIR = "vip"


def _q(s: str) -> str:
    return json.dumps(s)


def _py(obj) -> str:
    """A Python literal (not JSON: true/false/null are not Python)."""
    return repr(json.loads(json.dumps(obj, sort_keys=True, default=str)))


def _py_list(items) -> str:
    return "[" + ", ".join(_q(str(i)) for i in items) + "]"


# --------------------------------------------------------------------------
# Python (cocotb) template
# --------------------------------------------------------------------------

_HEADER = '''"""Interface VIP for edge {edge_id} (family {family}) -- GENERATED, do not edit.

Rendered from the frozen interface contract by orchestrator/langgraph/vip_lib.
Both the producer ({producer}) and the consumer ({consumer}) testbenches import
this module; it is the ONLY legitimate model of the other side of the edge.

Usage in a block testbench (cocotb):

    from vip.{module} import Driver, Monitor, Scoreboard, assertions, SIDES
    side = SIDES["consumer"]            # the DUT is the consumer of this edge
    drv = Driver(dut, side)            # drives the DUT's INPUT ports of the edge
    mon = Monitor(dut, side).start()   # observes accepted beats on the edge
    sb  = Scoreboard()
    cocotb.start_soon(assertions(dut, side))   # timing rules from contract.timing
    await drv.reset()
    await drv.send({{"field": value, ...}})    # or drv.request(...) for req_resp
    beat = await mon.next()

Sampling convention: drivers change inputs just after a rising edge; monitors
and assertions sample at the falling edge (stable current-cycle values); a
request accepted in cycle t and answered in cycle t+N has latency N.

{summary}
"""
from __future__ import annotations

import cocotb
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

EDGE_ID = {edge_id_q}
FAMILY = {family_q}
FINGERPRINT = {fp_q}
TIMING = {timing}
FIELDS = {fields}
PAYLOAD = {payload}
RESPONSE = {response}
ENUMS = {enums}
VALID = {valid_q}
READY = {ready_q}
RSP_VALID = {rsp_valid_q}
LAST = {last_q}
CLK = "clk"
RST = "rst_n"
SIDES = {sides}


class VIPError(AssertionError):
    """A contract rule was violated on this edge."""


def _sig(dut, side, name):
    port = side["ports"].get(name, name)
    if not hasattr(dut, port):
        raise VIPError(f"{{EDGE_ID}}: DUT has no port {{port!r}} for contract signal {{name!r}} "
                       f"(side={{side['role']}}, channel={{side['channel']!r}})")
    return getattr(dut, port)


def _has(dut, side, name):
    return bool(name) and hasattr(dut, side["ports"].get(name, name))


def _val(sig):
    try:
        return int(sig.value)
    except (ValueError, TypeError):
        return None


def _clk(dut):
    return getattr(dut, CLK)


def _in_reset(dut):
    rst = getattr(dut, RST, None)
    if rst is None:
        rst = getattr(dut, "rst", None)
        return bool(_val(rst)) if rst is not None else False
    v = _val(rst)
    return v == 0 if v is not None else True


async def _cycle_start(dut):
    """Just after a rising edge: safe to change inputs for the new cycle."""
    await RisingEdge(_clk(dut))
    await Timer(1, "step")


async def _sample(dut):
    """Mid-cycle sampling point (falling edge, read-only phase)."""
    await FallingEdge(_clk(dut))
    await ReadOnly()


def _ready_ok(dut, side):
    return (READY is None) or (not _has(dut, side, READY)) or _val(_sig(dut, side, READY)) == 1


def _payload(dut, side, names):
    return {{name: _val(_sig(dut, side, name)) for name in names if _has(dut, side, name)}}


class Scoreboard:
    """In-order expected-vs-observed comparison of beats (dicts)."""

    def __init__(self):
        self.expected = []
        self.observed = []
        self.mismatches = []

    def expect(self, beat):
        self.expected.append(dict(beat))

    def observe(self, beat):
        self.observed.append(dict(beat))
        i = len(self.observed) - 1
        if i < len(self.expected):
            exp = self.expected[i]
            diff = {{k: (exp[k], beat.get(k)) for k in exp if exp[k] is not None and beat.get(k) != exp[k]}}
            if diff:
                self.mismatches.append((i, diff))

    def check(self):
        if self.mismatches:
            raise VIPError(f"{{EDGE_ID}}: {{len(self.mismatches)}} beat mismatch(es); first: {{self.mismatches[0]}}")
        if len(self.observed) < len(self.expected):
            raise VIPError(f"{{EDGE_ID}}: expected {{len(self.expected)}} beats, observed {{len(self.observed)}}")
'''

_STREAM = '''

class Driver:
    """Producer-side driver of a {family} edge: presents a beat and holds it
    until accepted (valid_hold_until_ready), honouring the DUT's ready.
    When the DUT is the producer it drives READY instead (``Driver.ready``)."""

    def __init__(self, dut, side):
        self.dut, self.side = dut, side
        self.sent = []

    async def reset(self):
        await _cycle_start(self.dut)
        if self.side["role"] == "consumer":
            _sig(self.dut, self.side, VALID).value = 0
            for name in PAYLOAD:
                if _has(self.dut, self.side, name):
                    _sig(self.dut, self.side, name).value = 0
        elif READY and _has(self.dut, self.side, READY):
            _sig(self.dut, self.side, READY).value = 1

    async def send(self, beat, last=False, timeout_cycles=10000):
        """Present ``beat`` ({{signal: int}}) and hold it until accepted."""
        assert self.side["role"] == "consumer", "send() drives the DUT's input side"
        dut, side = self.dut, self.side
        await _cycle_start(dut)
        for name, v in beat.items():
            if _has(dut, side, name):
                _sig(dut, side, name).value = int(v)
        if LAST and _has(dut, side, LAST):
            _sig(dut, side, LAST).value = int(bool(last))
        _sig(dut, side, VALID).value = 1
        for _ in range(timeout_cycles):
            await _sample(dut)
            if _ready_ok(dut, side):
                break
            await RisingEdge(_clk(dut))
        else:
            raise VIPError(f"{{EDGE_ID}}: beat not accepted within {{timeout_cycles}} cycles")
        await _cycle_start(dut)
        _sig(dut, side, VALID).value = 0
        self.sent.append(dict(beat))

    async def ready(self, pattern=None, cycles=None):
        """Consumer-side ready driver (when the DUT is the producer):
        ``pattern`` is an iterable of 0/1 per cycle (None = always ready)."""
        assert READY and _has(self.dut, self.side, READY)
        sig = _sig(self.dut, self.side, READY)
        if pattern is None:
            await _cycle_start(self.dut)
            sig.value = 1
            return
        it = iter(pattern)
        n = 0
        while cycles is None or n < cycles:
            await _cycle_start(self.dut)
            try:
                sig.value = int(next(it))
            except StopIteration:
                sig.value = 1
                return
            n += 1


class Monitor:
    """Observes accepted beats (valid && ready, sampled mid-cycle)."""

    def __init__(self, dut, side, scoreboard=None):
        self.dut, self.side, self.sb = dut, side, scoreboard
        self.beats = []
        self._task = None

    def start(self):
        self._task = cocotb.start_soon(self._run())
        return self

    async def _run(self):
        dut, side = self.dut, self.side
        while True:
            await _sample(dut)
            if _in_reset(dut):
                continue
            if _val(_sig(dut, side, VALID)) == 1 and _ready_ok(dut, side):
                beat = _payload(dut, side, PAYLOAD)
                if LAST and _has(dut, side, LAST):
                    beat["__last"] = _val(_sig(dut, side, LAST))
                self.beats.append(beat)
                if self.sb is not None:
                    self.sb.observe(beat)

    async def next(self, timeout_cycles=10000):
        n = len(self.beats)
        for _ in range(timeout_cycles):
            await RisingEdge(_clk(self.dut))
            if len(self.beats) > n:
                return self.beats[n]
        raise VIPError(f"{{EDGE_ID}}: no beat within {{timeout_cycles}} cycles")


async def assertions(dut, side):
    """Contract timing rules, checked every cycle. Raise VIPError on violation:
      * valid_hold_until_ready: valid && !ready must keep valid AND the payload next cycle
      * valid_to_ready_max_stall: ready within N cycles of valid
      * reset_idle_cycles: valid low for N cycles after reset deasserts
    """
    hold = bool(TIMING.get("valid_hold_until_ready"))
    max_stall = TIMING.get("valid_to_ready_max_stall")
    idle = int(TIMING.get("reset_idle_cycles") or 0)
    prev_valid = prev_ready = 0
    prev_payload = None
    stall = 0
    since_reset = None
    while True:
        await _sample(dut)
        if _in_reset(dut):
            since_reset = 0
            prev_valid = 0
            stall = 0
            continue
        v = _val(_sig(dut, side, VALID)) or 0
        r = 1 if _ready_ok(dut, side) else 0
        payload = tuple(_val(_sig(dut, side, name)) for name in PAYLOAD if _has(dut, side, name))
        if since_reset is not None:
            if since_reset < idle and v:
                raise VIPError(f"{{EDGE_ID}}: {{VALID}} high {{since_reset}} cycle(s) after reset; "
                               f"contract reset_idle_cycles={{idle}}")
            since_reset += 1
        if hold and prev_valid and not prev_ready:
            if not v:
                raise VIPError(f"{{EDGE_ID}}: {{VALID}} dropped while not accepted (valid_hold_until_ready)")
            if payload != prev_payload:
                raise VIPError(f"{{EDGE_ID}}: payload changed while valid && !ready (valid_hold_until_ready)")
        if v and not r:
            stall += 1
            if max_stall is not None and stall > int(max_stall):
                raise VIPError(f"{{EDGE_ID}}: ready withheld {{stall}} cycles > valid_to_ready_max_stall={{max_stall}}")
        else:
            stall = 0
        prev_valid, prev_ready, prev_payload = v, r, payload
'''

_REQ_RESP = '''

class Driver:
    """Requester-side driver of a req_resp edge (DUT is the responder), or the
    responder model when the DUT is the requester (``respond()``)."""

    def __init__(self, dut, side):
        self.dut, self.side = dut, side
        self.requests = []
        self.responses = []

    async def reset(self):
        await _cycle_start(self.dut)
        if self.side["role"] == "consumer":
            _sig(self.dut, self.side, VALID).value = 0
            for name in PAYLOAD:
                if _has(self.dut, self.side, name):
                    _sig(self.dut, self.side, name).value = 0
        else:
            if _has(self.dut, self.side, RSP_VALID):
                _sig(self.dut, self.side, RSP_VALID).value = 0
            if READY and _has(self.dut, self.side, READY):
                _sig(self.dut, self.side, READY).value = 1

    async def request(self, req, timeout_cycles=10000):
        """Issue one request, wait for acceptance, then for the response
        inside the contract latency window. Returns the response dict
        (with ``__latency`` = cycles after the accept cycle)."""
        assert self.side["role"] == "consumer", "request() drives the DUT's request port"
        dut, side = self.dut, self.side
        await _cycle_start(dut)
        for name, v in req.items():
            if _has(dut, side, name):
                _sig(dut, side, name).value = int(v)
        _sig(dut, side, VALID).value = 1
        accepted = False
        for _ in range(timeout_cycles):
            await _sample(dut)
            if _ready_ok(dut, side):
                accepted = True
                break
            await RisingEdge(_clk(dut))
        if not accepted:
            raise VIPError(f"{{EDGE_ID}}: request not accepted within {{timeout_cycles}} cycles")
        self.requests.append(dict(req))
        lat = TIMING.get("req_to_rsp_cycles") or {{}}
        lo, hi = lat.get("min"), lat.get("max")
        limit = int(hi) if hi is not None else timeout_cycles
        # cycle t = the accept cycle; the next mid-cycle sample is cycle t+1
        await _cycle_start(dut)
        _sig(dut, side, VALID).value = 0
        waited = 1
        while True:
            await _sample(dut)
            if _has(dut, side, RSP_VALID) and _val(_sig(dut, side, RSP_VALID)) == 1:
                if lo is not None and waited < int(lo):
                    raise VIPError(f"{{EDGE_ID}}: response after {{waited}} cycle(s) < contract min {{lo}}")
                rsp = _payload(dut, side, RESPONSE)
                rsp["__latency"] = waited
                self.responses.append(rsp)
                return rsp
            if waited >= limit:
                raise VIPError(f"{{EDGE_ID}}: no response after {{waited}} cycle(s); contract max {{hi}}")
            await RisingEdge(_clk(dut))
            waited += 1

    async def respond(self, handler, latency=None):
        """Responder model when the DUT is the requester: on each accepted
        request call ``handler(req) -> rsp dict`` and drive it ``latency``
        cycles later (default: the contract's exact/min)."""
        assert self.side["role"] == "producer"
        dut, side = self.dut, self.side
        lat = TIMING.get("req_to_rsp_cycles") or {{}}
        n = latency if latency is not None else (lat.get("exact") if lat.get("exact") is not None else (lat.get("min") or 1))
        pending = []
        while True:
            await _sample(dut)
            if not _in_reset(dut) and _val(_sig(dut, side, VALID)) == 1 and _ready_ok(dut, side):
                pending.append([int(n), handler(_payload(dut, side, PAYLOAD))])
            await _cycle_start(dut)
            for p in pending:
                p[0] -= 1
            fire = [p for p in pending if p[0] <= 0]
            pending = [p for p in pending if p[0] > 0]
            if _has(dut, side, RSP_VALID):
                _sig(dut, side, RSP_VALID).value = 1 if fire else 0
            if fire:
                for name, val in (fire[0][1] or {{}}).items():
                    if _has(dut, side, name):
                        _sig(dut, side, name).value = int(val)


class Monitor:
    """Observes accepted requests and responses on the edge (mid-cycle)."""

    def __init__(self, dut, side, scoreboard=None):
        self.dut, self.side, self.sb = dut, side, scoreboard
        self.requests = []
        self.responses = []
        self.latencies = []
        self._task = None

    def start(self):
        self._task = cocotb.start_soon(self._run())
        return self

    async def _run(self):
        dut, side = self.dut, self.side
        outstanding = []
        while True:
            await _sample(dut)
            if _in_reset(dut):
                outstanding = []
                continue
            accept = _val(_sig(dut, side, VALID)) == 1 and _ready_ok(dut, side)
            for o in outstanding:
                o[0] += 1
            if _has(dut, side, RSP_VALID) and _val(_sig(dut, side, RSP_VALID)) == 1:
                rsp = _payload(dut, side, RESPONSE)
                self.responses.append(rsp)
                if outstanding:
                    self.latencies.append(outstanding.pop(0)[0])
                if self.sb is not None:
                    self.sb.observe(rsp)
            if accept:
                req = _payload(dut, side, PAYLOAD)
                self.requests.append(req)
                outstanding.append([0, req])

    async def next(self, timeout_cycles=10000):
        n = len(self.responses)
        for _ in range(timeout_cycles):
            await RisingEdge(_clk(self.dut))
            if len(self.responses) > n:
                return self.responses[n]
        raise VIPError(f"{{EDGE_ID}}: no response within {{timeout_cycles}} cycles")


async def assertions(dut, side):
    """Contract timing rules, checked every cycle:
      * req_to_rsp_cycles: every accepted request answers within [min, max]
        cycles (exact when the contract says so); a response with no
        outstanding request is a violation (in_order)
      * reset_idle_cycles: request valid low for N cycles after reset
    """
    lat = TIMING.get("req_to_rsp_cycles") or {{}}
    lo, hi = lat.get("min"), lat.get("max")
    idle = int(TIMING.get("reset_idle_cycles") or 0)
    outstanding = []
    since_reset = None
    while True:
        await _sample(dut)
        if _in_reset(dut):
            outstanding = []
            since_reset = 0
            continue
        v = _val(_sig(dut, side, VALID)) or 0
        accept = bool(v) and _ready_ok(dut, side)
        rv = (_val(_sig(dut, side, RSP_VALID)) or 0) if _has(dut, side, RSP_VALID) else 0
        if since_reset is not None:
            if since_reset < idle and v:
                raise VIPError(f"{{EDGE_ID}}: {{VALID}} high {{since_reset}} cycle(s) after reset; "
                               f"contract reset_idle_cycles={{idle}}")
            since_reset += 1
        outstanding = [age + 1 for age in outstanding]
        if rv:
            if not outstanding:
                raise VIPError(f"{{EDGE_ID}}: {{RSP_VALID}} with no outstanding request")
            age = outstanding.pop(0)
            if lo is not None and age < int(lo):
                raise VIPError(f"{{EDGE_ID}}: response after {{age}} cycle(s) < contract min {{lo}}")
            if hi is not None and age > int(hi):
                raise VIPError(f"{{EDGE_ID}}: response after {{age}} cycle(s) > contract max {{hi}}")
        for age in outstanding:
            if hi is not None and age > int(hi):
                raise VIPError(f"{{EDGE_ID}}: request unanswered for {{age}} cycle(s) > contract max {{hi}}")
        if accept:
            outstanding.append(0)
'''

_VALID_ONLY = '''

class Driver:
    """Drives a {family} edge (always accepted: no ready)."""

    def __init__(self, dut, side):
        self.dut, self.side = dut, side
        self.sent = []

    async def reset(self):
        await _cycle_start(self.dut)
        if self.side["role"] == "consumer":
            if VALID and _has(self.dut, self.side, VALID):
                _sig(self.dut, self.side, VALID).value = 0
            for name in PAYLOAD:
                if _has(self.dut, self.side, name):
                    _sig(self.dut, self.side, name).value = 0

    async def send(self, beat, hold_cycles=1):
        assert self.side["role"] == "consumer", "send() drives the DUT's input side"
        dut, side = self.dut, self.side
        await _cycle_start(dut)
        for name, v in beat.items():
            if _has(dut, side, name):
                _sig(dut, side, name).value = int(v)
        if VALID and _has(dut, side, VALID):
            _sig(dut, side, VALID).value = 1
        for _ in range(hold_cycles):
            await _cycle_start(dut)
        if VALID and _has(dut, side, VALID):
            _sig(dut, side, VALID).value = 0
        self.sent.append(dict(beat))


class Monitor:
    """Observes every cycle the strobe is high (or every change, for static)."""

    def __init__(self, dut, side, scoreboard=None):
        self.dut, self.side, self.sb = dut, side, scoreboard
        self.beats = []
        self._task = None

    def start(self):
        self._task = cocotb.start_soon(self._run())
        return self

    async def _run(self):
        dut, side = self.dut, self.side
        prev = None
        while True:
            await _sample(dut)
            if _in_reset(dut):
                continue
            beat = _payload(dut, side, PAYLOAD)
            if VALID and _has(dut, side, VALID):
                if _val(_sig(dut, side, VALID)) == 1:
                    self.beats.append(beat)
                    if self.sb is not None:
                        self.sb.observe(beat)
            elif beat != prev:
                self.beats.append(beat)
                if self.sb is not None:
                    self.sb.observe(beat)
                prev = beat

    async def next(self, timeout_cycles=10000):
        n = len(self.beats)
        for _ in range(timeout_cycles):
            await RisingEdge(_clk(self.dut))
            if len(self.beats) > n:
                return self.beats[n]
        raise VIPError(f"{{EDGE_ID}}: no beat within {{timeout_cycles}} cycles")


async def assertions(dut, side):
    """Contract rules for an always-accepted edge:
      * reset_idle_cycles: the strobe is low N cycles after reset
      * no X on the strobe after reset
    """
    idle = int(TIMING.get("reset_idle_cycles") or 0)
    since_reset = None
    while True:
        await _sample(dut)
        if _in_reset(dut):
            since_reset = 0
            continue
        if VALID and _has(dut, side, VALID):
            v = _val(_sig(dut, side, VALID))
            if v is None:
                raise VIPError(f"{{EDGE_ID}}: {{VALID}} is X/Z after reset")
            if since_reset is not None and since_reset < idle and v:
                raise VIPError(f"{{EDGE_ID}}: {{VALID}} high {{since_reset}} cycle(s) after reset; "
                               f"contract reset_idle_cycles={{idle}}")
        if since_reset is not None:
            since_reset += 1
'''



from .bus_template import BUS_TEMPLATE as _BUS  # noqa: E402


def render_vip_module(edge: dict) -> str:
    ec = EdgeContract.from_contract(edge)
    from orchestrator.architecture.specialists.contract_timing import timing_summary
    sides = {}
    for role in ("producer", "consumer"):
        sd = ec.side(role)
        sides[role] = {"block": sd.block, "role": role, "channel": sd.channel, "ports": sd.ports}
    body = _HEADER.format(
        edge_id=ec.edge_id, family=ec.family, producer=ec.producer, consumer=ec.consumer,
        module=ec.module_name, summary=timing_summary({"timing": ec.timing}),
        edge_id_q=_q(ec.edge_id), family_q=_q(ec.family), fp_q=_q(ec.fingerprint()),
        timing=_py(ec.timing),
        fields=_py([{"name": f.get("name"), "width": f.get("width"), "msb": f.get("msb"),
                     "lsb": f.get("lsb")} for f in ec.fields if isinstance(f, dict)]),
        payload=_py_list([s.name for s in ec.payload_signals]),
        response=_py_list([s.name for s in ec.response_signals]),
        enums=_py(ec.enums),
        valid_q=_q(ec.valid_signal) if ec.valid_signal else "None",
        ready_q=_q(ec.ready_signal) if ec.ready_signal else "None",
        rsp_valid_q=_q(ec.response_valid_signal) if ec.response_valid_signal else "None",
        last_q=_q(ec.last_signal) if ec.last_signal else "None",
        sides=_py(sides),
    )
    if ec.family in ("axi_stream", "srdy_drdy"):
        body += _STREAM.format(family=ec.family)
    elif ec.family == "req_resp":
        body += _REQ_RESP.format()  # un-double the braces; no fields to fill
    elif ec.family in ("axi4", "axi_lite", "apb"):
        from orchestrator.fabric.amba import HANDSHAKES
        body += f"HANDSHAKES = {HANDSHAKES[ec.family]!r}\n" + _BUS
    else:
        body += _VALID_ONLY.format(family=ec.family)
    return body


# --------------------------------------------------------------------------
# SVA bind ($past-based: no cycle-delay operators, so plain `verilator --assert`)
# --------------------------------------------------------------------------

def render_sva_bind(edge: dict, dut_module: str, role: str) -> str:
    """A bindable checker for the DUT on ``role`` side of the edge.

    Every property is written with ``$past`` instead of cycle-delay
    operators, so it compiles under plain ``verilator --assert`` (no
    ``--timing``) and on any simulator; the Python ``assertions()`` coroutine
    remains the oracle for rules that need unbounded history.
    """
    ec = EdgeContract.from_contract(edge)
    sd = ec.side(role)
    name = f"{ec.module_name}__{role}_chk"
    t = ec.timing
    ports: list[str] = ["input wire clk", "input wire rst_n"]
    props: list[str] = []
    v = ec.valid_signal
    r = ec.ready_signal
    pv = sd.port(v) if v and v in sd.ports else None
    pr = sd.port(r) if r and r in sd.ports else None
    if pv:
        ports.append(f"input wire {pv}")
        idle = int(t.get("reset_idle_cycles") or 0)
        if idle >= 1:
            props.append(
                f"  // reset_idle_cycles={idle}: valid low right after reset deasserts\n"
                f"  a_reset_idle: assert property (@(posedge clk) "
                f"($past(!rst_n) && rst_n) |-> !{pv});")
    if ec.family in ("axi_stream", "srdy_drdy") and pv and pr:
        ports.append(f"input wire {pr}")
        if t.get("valid_hold_until_ready"):
            props.append(
                f"  // valid_hold_until_ready: a presented beat stays until accepted\n"
                f"  a_hold: assert property (@(posedge clk) disable iff (!rst_n) "
                f"$past({pv} && !{pr}) |-> {pv});")
        ms = t.get("valid_to_ready_max_stall")
        if ms is not None and 1 <= int(ms) <= 32:
            n = int(ms)
            terms = " && ".join([f"({pv} && !{pr})"] + [f"$past({pv} && !{pr}, {k})" for k in range(1, n + 1)])
            props.append(
                f"  // valid_to_ready_max_stall={n}: never {n + 1} consecutive stalled cycles\n"
                f"  a_stall: assert property (@(posedge clk) disable iff (!rst_n) !({terms}));")
    if ec.family == "req_resp":
        rv = ec.response_valid_signal
        lat = t.get("req_to_rsp_cycles") or {}
        if rv and rv in sd.ports and pv:
            prv = sd.port(rv)
            ports.append(f"input wire {prv}")
            if pr:
                ports.append(f"input wire {pr}")
            accept = f"({pv} && {pr})" if pr else pv
            if lat.get("exact") is not None and int(lat["exact"]) >= 1:
                n = int(lat["exact"])
                props.append(
                    f"  // req_to_rsp_cycles exact={n}\n"
                    f"  a_latency: assert property (@(posedge clk) disable iff (!rst_n) "
                    f"$past({accept}, {n}) |-> {prv});")
            elif lat.get("max") is not None and int(lat["max"]) >= 1:
                lo, hi = max(0, int(lat.get("min") or 0)), int(lat["max"])
                window = " || ".join([prv] + [f"$past({prv}, {k})" for k in range(1, hi - lo + 1)])
                props.append(
                    f"  // req_to_rsp_cycles [{lo}:{hi}]: a response lands within the window\n"
                    f"  a_latency: assert property (@(posedge clk) disable iff (!rst_n) "
                    f"$past({accept}, {hi}) |-> ({window}));")
    seen, uports = set(), []
    for p in ports:
        if p not in seen:
            seen.add(p)
            uports.append(p)
    lines = [
        f"// GENERATED interface checker for edge {ec.edge_id} ({role} side of {sd.block}).",
        "// Rendered from the frozen contract's timing object; do not edit.",
        "// $past-based (no cycle-delay operators): compiles under `verilator --assert` without --timing.",
        "`ifndef SYNTHESIS",
        "// coverage_off: verilator 5.051 --coverage crashes on $past in bound checkers",
        "// (V3Localize.cpp:203 'AstVarRef not under function'); DUT coverage is unaffected.",
        "/* verilator coverage_off */",
        f"module {name} (",
        "  " + ",\n  ".join(uports),
        ");",
    ]
    lines += props or ["  // no bindable rules for this family/timing"]
    lines += ["endmodule", "/* verilator coverage_on */", ""]
    bind_ports = ", ".join(f".{p.split()[-1]}({p.split()[-1]})" for p in uports)
    lines += [f"bind {dut_module} {name} u_{name} ({bind_ports});", "`endif", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Files and index
# --------------------------------------------------------------------------

def _vip_dir(project_root) -> Path:
    return Path(project_root) / ".coresmith" / VIP_DIR


def write_vip(project_root, edge: dict, *, dut_modules: dict[str, str] | None = None) -> dict:
    """Write ``<edge>.py`` and the two ``<edge>__<role>_sva.sv`` files."""
    ec = EdgeContract.from_contract(edge)
    d = _vip_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text("# generated interface VIPs (orchestrator/langgraph/vip_lib)\n")
    py = d / f"{ec.module_name}.py"
    py.write_text(render_vip_module(edge))
    sva = {}
    for role in ("producer", "consumer"):
        sd = ec.side(role)
        mod = (dut_modules or {}).get(sd.block) or sd.block
        f = d / f"{ec.module_name}__{role}_sva.sv"
        f.write_text(render_sva_bind(edge, mod, role))
        sva[role] = str(f)
    return {"edge_id": ec.edge_id, "module": ec.module_name, "family": ec.family,
            "py": str(py), "sva": sva, "fingerprint": ec.fingerprint(),
            "producer": ec.producer, "consumer": ec.consumer,
            "producer_channel": ec.side("producer").channel,
            "consumer_channel": ec.side("consumer").channel}


def write_all_vips(project_root, contracts: list[dict], *, contract_version=None,
                   dut_modules: dict[str, str] | None = None) -> dict:
    """Render every edge and write ``vip_index.json``."""
    index = {"generated_at": time.time(), "contract_version": contract_version, "edges": {}}
    for c in contracts or []:
        if not isinstance(c, dict):
            continue
        try:
            info = write_vip(project_root, c, dut_modules=dut_modules)
        except Exception as exc:  # noqa: BLE001 - one bad edge must not lose the rest
            info = {"edge_id": str(c.get("edge_id")), "error": str(exc)}
        index["edges"][info["edge_id"]] = info
    d = _vip_dir(project_root)
    d.mkdir(parents=True, exist_ok=True)
    (d.parent / "vip_index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
    return index


def vip_index(project_root) -> dict:
    p = Path(project_root) / ".coresmith" / "vip_index.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {"edges": {}}


def vips_for_block(project_root, block: str) -> list[dict]:
    """``[{edge_id, module, role, py, sva, channel, family}]`` for one block."""
    out = []
    for eid, info in (vip_index(project_root).get("edges") or {}).items():
        if info.get("error"):
            continue
        for role in ("producer", "consumer"):
            if info.get(role) == block:
                out.append({"edge_id": eid, "module": info["module"], "role": role,
                            "py": info["py"], "sva": (info.get("sva") or {}).get(role),
                            "channel": info.get(f"{role}_channel"), "family": info.get("family")})
    return out


def vip_enabled() -> bool:
    return (os.environ.get("CORESMITH_INTERFACE_VIP", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off", ""}


def sva_bind_enabled() -> bool:
    return (os.environ.get("CORESMITH_VIP_SVA_BIND", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off", ""}
