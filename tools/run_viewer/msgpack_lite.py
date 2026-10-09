"""Just enough MessagePack to read a LangGraph checkpoint header.

LangGraph's SQLite saver stores each checkpoint as msgpack; the viewer needs
only its ``ts`` and the node names in ``versions_seen``. Extension types
(LangGraph's serialised objects) are returned as opaque ``Ext`` markers and
never interpreted.
"""
from __future__ import annotations

import struct


class Ext:
    __slots__ = ("code", "size")

    def __init__(self, code: int, size: int):
        self.code, self.size = code, size

    def __repr__(self) -> str:
        return f"Ext({self.code}, {self.size}B)"


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.d):
            raise ValueError("truncated msgpack")
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def u(self, fmt: str, n: int):
        return struct.unpack(fmt, self.take(n))[0]

    def obj(self):
        t = self.take(1)[0]
        if t <= 0x7F:
            return t
        if 0x80 <= t <= 0x8F:
            return self.map(t & 0x0F)
        if 0x90 <= t <= 0x9F:
            return self.arr(t & 0x0F)
        if 0xA0 <= t <= 0xBF:
            return self.str(t & 0x1F)
        if t >= 0xE0:
            return t - 0x100
        if t == 0xC0:
            return None
        if t == 0xC2:
            return False
        if t == 0xC3:
            return True
        if t in (0xC4, 0xC5, 0xC6):
            n = self.u({0xC4: ">B", 0xC5: ">H", 0xC6: ">I"}[t], {0xC4: 1, 0xC5: 2, 0xC6: 4}[t])
            return bytes(self.take(n))
        if t in (0xC7, 0xC8, 0xC9):
            n = self.u({0xC7: ">B", 0xC8: ">H", 0xC9: ">I"}[t], {0xC7: 1, 0xC8: 2, 0xC9: 4}[t])
            code = self.u(">b", 1)
            self.take(n)
            return Ext(code, n)
        if t == 0xCA:
            return self.u(">f", 4)
        if t == 0xCB:
            return self.u(">d", 8)
        ints = {0xCC: (">B", 1), 0xCD: (">H", 2), 0xCE: (">I", 4), 0xCF: (">Q", 8),
                0xD0: (">b", 1), 0xD1: (">h", 2), 0xD2: (">i", 4), 0xD3: (">q", 8)}
        if t in ints:
            return self.u(*ints[t])
        if t in (0xD4, 0xD5, 0xD6, 0xD7, 0xD8):
            n = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}[t]
            code = self.u(">b", 1)
            self.take(n)
            return Ext(code, n)
        if t in (0xD9, 0xDA, 0xDB):
            n = self.u({0xD9: ">B", 0xDA: ">H", 0xDB: ">I"}[t], {0xD9: 1, 0xDA: 2, 0xDB: 4}[t])
            return self.str(n)
        if t in (0xDC, 0xDD):
            return self.arr(self.u(">H" if t == 0xDC else ">I", 2 if t == 0xDC else 4))
        if t in (0xDE, 0xDF):
            return self.map(self.u(">H" if t == 0xDE else ">I", 2 if t == 0xDE else 4))
        raise ValueError(f"unsupported msgpack type 0x{t:02x}")

    def str(self, n: int) -> str:
        return self.take(n).decode("utf-8", "replace")

    def arr(self, n: int) -> list:
        return [self.obj() for _ in range(n)]

    def map(self, n: int) -> dict:
        out = {}
        for _ in range(n):
            k = self.obj()
            out[k if isinstance(k, (str, int, float, bool)) or k is None else repr(k)] = self.obj()
        return out


def unpack(data: bytes):
    return _Reader(bytes(data)).obj()


def checkpoint_header(blob: bytes) -> dict:
    """``{ts, seen, channels}`` of a LangGraph checkpoint blob: ``seen`` maps
    each node to a digest of the channel versions it has consumed, so the
    nodes that ran between a checkpoint and its parent are the ones whose
    digest changed. Channel values are never returned. Errors are reported,
    not raised."""
    try:
        cp = unpack(blob)
    except (ValueError, struct.error, IndexError) as exc:
        return {"ts": None, "seen": {}, "error": f"undecodable checkpoint: {exc}"}
    if not isinstance(cp, dict):
        return {"ts": None, "seen": {}, "error": "checkpoint is not a map"}
    seen = cp.get("versions_seen") if isinstance(cp.get("versions_seen"), dict) else {}
    chans = cp.get("channel_versions") if isinstance(cp.get("channel_versions"), dict) else {}
    digest = {}
    for node, versions in seen.items():
        if str(node).startswith("__"):
            continue
        items = sorted((str(k), repr(v)) for k, v in versions.items()) if isinstance(versions, dict) else [repr(versions)]
        digest[str(node)] = repr(items)
    return {"ts": cp.get("ts") if isinstance(cp.get("ts"), str) else None, "seen": digest, "channels": len(chans)}
