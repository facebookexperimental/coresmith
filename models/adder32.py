"""Golden model for the CoreSmith adder32 smoke design."""

from __future__ import annotations


def add32(a: int, b: int, cin: int = 0) -> tuple[int, int]:
    """Return (sum, cout) for 32-bit unsigned addition."""
    total = (int(a) & 0xFFFFFFFF) + (int(b) & 0xFFFFFFFF) + (int(cin) & 1)
    return total & 0xFFFFFFFF, (total >> 32) & 1


def reference(a: int, b: int, cin: int = 0) -> dict[str, int]:
    """Dictionary form used by generated cocotb/validation code."""
    sum_value, cout = add32(a, b, cin)
    return {"sum": sum_value, "cout": cout}
