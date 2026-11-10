"""The per-expert routing counter's file, as the cockpit reads it (dashboard/cockpit.py).

The flash-local lane's engine mounts flash-local-experts/ (this repo) as a vLLM plugin that
counts, per MoE layer, the tokens routed to each expert, and writes the matrix twice a second
to /dev/shm/qwen38-experts.bin: the unit runs with --ipc=host, so the host's /dev/shm is the
container's and the cockpit reads the file in place, no port, no request to the engine. The
format is flash_local_experts/wire.py's, repeated here in the standard library alone so the
cockpit imports nothing of the engine's side: a 48-byte header, a JSON block with the model's
shape, then int32 counts, rows x experts. The two stay one format by the test that writes
with the plugin and reads with this.
"""
from __future__ import annotations

import json
import os
import struct
import time

MAGIC = b"QEXP"
VERSION = 1
HEADER = struct.Struct("<4sIIIIIQdII")


def pad4(n: int) -> int:
    return (n + 3) & ~3


def parse_header(data: bytes) -> dict | None:
    """The header and the JSON block, from the file's first bytes; None for any other file."""
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
            "meta": meta, "meta_len": meta_len}


def whole(data: bytes) -> dict | None:
    """The header of a file that is whole: the counts are there to their last byte."""
    h = parse_header(data)
    if h is None or len(data) != HEADER.size + pad4(h["meta_len"]) + h["rows"] * h["experts"] * 4:
        return None
    return h


def read_header(path: str, now: float | None = None) -> dict:
    """What the Experts view's rail entry and caption need, read every few seconds: whether a
    counter writes, how old its last write is, and the model's shape. Never raises: a file
    that is not there, or not this format, is {"present": False}."""
    now = time.time() if now is None else now
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            head = f.read(HEADER.size + 65536)
    except OSError:
        return {"present": False, "path": path}
    h = parse_header(head)
    if h is None:
        return {"present": False, "path": path, "unreadable": True}
    meta = h["meta"]
    return {"present": True, "path": path, "age": max(0.0, now - st.st_mtime), "wall": h["wall"],
            "seq": h["seq"], "pid": h["pid"], "rows": h["rows"], "experts": h["experts"],
            "top_k": h["top_k"], "model": meta.get("served_model_name") or meta.get("model") or "",
            "model_type": meta.get("model_type") or "", "spec_method": meta.get("spec_method")}


def read_frame(path: str) -> tuple[bytes | None, float]:
    """The whole file for the view's poll, and its age in seconds; (None, 0) when there is no
    whole file of this format (a reader never sees a write in progress: the plugin renames a
    finished sibling over the path)."""
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None, 0.0
    if whole(data) is None:
        return None, 0.0
    return data, max(0.0, time.time() - st.st_mtime)
