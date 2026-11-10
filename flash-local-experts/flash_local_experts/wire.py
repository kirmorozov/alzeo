"""The counter file: what the plugin writes and the cockpit reads.

Little-endian, a 48-byte header, a JSON block, then the counts:

    0   4s  magic, b"QEXP"
    4   I   version, 1
    8   I   rows: MoE layers counted, in the order the engine built them
    12  I   experts per layer
    16  I   experts routed to per token (top_k)
    20  I   pid of the writer
    24  Q   seq: the writer's write number, 1 and up
    32  d   wall: the writer's clock at the write, Unix seconds
    40  I   meta_len: the JSON block's length in bytes
    44  I   reserved, 0
    48      the JSON block (meta_len bytes), zero-padded to a multiple of 4
            int32 counts, rows x experts, row-major: tokens routed to that expert since the
            engine started

The JSON block is the model's shape as the engine reads it, for the drawing (layer types,
sizes, the speculative method, the rows' layer names). Counts are cumulative: a reader keeps
the previous frame and draws the difference, so a frame missed costs nothing. The file is
written whole to a sibling and renamed over, so a reader never sees half of it. No numpy and
no torch here: the cockpit reads this with the standard library alone.
"""
from __future__ import annotations

import json
import struct

MAGIC = b"QEXP"
VERSION = 1
HEADER = struct.Struct("<4sIIIIIQdII")
assert HEADER.size == 48


def pad4(n: int) -> int:
    return (n + 3) & ~3


def pack(meta: dict, counts: bytes, rows: int, experts: int, top_k: int, seq: int, wall: float,
         pid: int) -> bytes:
    """One whole file. counts holds rows x experts little-endian int32, row-major."""
    body = json.dumps(meta, separators=(",", ":")).encode()
    if len(counts) != rows * experts * 4:
        raise ValueError(f"counts carry {len(counts)} bytes, not {rows} x {experts} x 4")
    head = HEADER.pack(MAGIC, VERSION, rows, experts, top_k, pid, seq, wall, len(body), 0)
    return head + body + b"\0" * (pad4(len(body)) - len(body)) + counts


def unpack(data: bytes) -> dict | None:
    """The file's header, meta and counts, or None for anything that is not a whole file of
    this format (a sibling mid-write is never seen, but a reader may be older or newer)."""
    if len(data) < HEADER.size:
        return None
    magic, version, rows, experts, top_k, pid, seq, wall, meta_len, _ = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        return None
    start = HEADER.size + pad4(meta_len)
    if len(data) != start + rows * experts * 4:
        return None
    try:
        meta = json.loads(data[HEADER.size:HEADER.size + meta_len].decode())
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    return {"rows": rows, "experts": experts, "top_k": top_k, "pid": pid, "seq": seq, "wall": wall,
            "meta": meta, "counts": memoryview(data)[start:]}


def unpack_header(data: bytes) -> dict | None:
    """The header and meta alone, from the file's first bytes (a reader that wants to know
    whether a counter is there and what it counts, without the counts)."""
    if len(data) < HEADER.size:
        return None
    magic, version, rows, experts, top_k, pid, seq, wall, meta_len, _ = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION or len(data) < HEADER.size + meta_len:
        return None
    try:
        meta = json.loads(data[HEADER.size:HEADER.size + meta_len].decode())
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    return {"rows": rows, "experts": experts, "top_k": top_k, "pid": pid, "seq": seq, "wall": wall,
            "meta": meta}
