"""Golden OpenFrame wrapper behavior for the adder32 backend smoke harness."""

from __future__ import annotations

from models.adder32 import add32


def openframe_gpio_reference(io_in: int) -> dict[str, int]:
    """Pure combinational GPIO mapping used by the generated wrapper.

    io_in[33:2] carries the data word, io_in[34] carries cin, and
    io_in[36:35] selects the operand mode:
      00: a=data, b=0
      01: a=0, b=data
      10: a=data, b=data
      11: a=data, b=~data
    """
    io_in = int(io_in) & ((1 << 44) - 1)
    data = (io_in >> 2) & 0xFFFFFFFF
    cin = (io_in >> 34) & 1
    mode = (io_in >> 35) & 0x3

    if mode == 0:
        a, b = data, 0
    elif mode == 1:
        a, b = 0, data
    elif mode == 2:
        a, b = data, data
    else:
        a, b = data, (~data) & 0xFFFFFFFF

    sum_value, cout = add32(a, b, cin)

    io_out = 0
    io_out |= (sum_value & 0xFFFFFFFF) << 2
    io_out |= (cout & 1) << 37

    result_enable = (mode >> 1) & 1
    io_oeb = (1 << 44) - 1
    if result_enable:
        io_oeb &= ~(((1 << 32) - 1) << 2)
    io_oeb &= ~(1 << 37)
    io_oeb &= ~(((1 << 5) - 1) << 39)

    return {
        "a": a,
        "b": b,
        "cin": cin,
        "sum": sum_value,
        "cout": cout,
        "io_out": io_out,
        "io_oeb": io_oeb & ((1 << 44) - 1),
        "result_enable": result_enable,
        "mode": mode,
    }


class Adder32WrapperModel:
    """Stateless adapter exposing the combinational GPIO reference."""

    def eval(self, io_in: int) -> dict[str, int]:
        return openframe_gpio_reference(io_in)
