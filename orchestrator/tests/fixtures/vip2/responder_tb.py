"""vip2 fixture testbench: drives the responder ONLY through the generated VIP."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from vip.req__m_q__to__responder__s_q import SIDES, Driver, Monitor, Scoreboard, assertions


async def _reset(dut):
    dut.rst_n.value = 0
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_requests_answered_per_contract(dut):
    side = SIDES["consumer"]
    drv = Driver(dut, side)
    sb = Scoreboard()
    mon = Monitor(dut, side, sb).start()
    cocotb.start_soon(assertions(dut, side))
    await _reset(dut)
    await drv.reset()
    for a in (1, 2, 250):
        sb.expect({"rdata": (a + 1) & 0xFF})
        rsp = await drv.request({"addr": a})
        assert rsp["rdata"] == (a + 1) & 0xFF, rsp
    for _ in range(4):
        await RisingEdge(dut.clk)
    sb.check()
    assert len(mon.requests) == 3 and mon.latencies == [1, 1, 1], (mon.requests, mon.latencies)
