"""The count: one row per MoE layer, one cell per expert, kept on the GPU and copied out.

Where it hooks. vLLM's MoE router (BaseRouter._select_experts) carries a capture_fn slot,
called with the step's topk_ids right after routing and before any EPLB mapping; the image's
own routed-experts feature (RoutedExpertsCapturer) binds there. This plugin binds a counter
there at each MoERunner's construction (plugin.py), one row per runner in construction order:
the 48 decoder layers, then the MTP head's own layer when the unit drafts with it. The rows'
layer names go into the file's meta, so the reader tells the two apart by name.

The hot path, per MoE layer per step: one index_add_ of a slice of a preallocated ones vector
into the row of a preallocated int32 matrix, no allocation, no host sync; the row 0 call also
queues an asynchronous device-to-host copy of the whole matrix into pinned memory. Both are
CUDA kernels, so a CUDA graph captured over a step records and replays them. The MoE runs
inside the torch.ops.vllm.moe_forward custom op, which torch.compile treats as opaque: the
slot runs as eager Python inside it, and the compiled graph and its cached AOT artifact are
unchanged. A batch wider than the ones vector (none is: it is twice the scheduler's ceiling)
is left uncounted rather than grown on the hot path.

Padding counts too: a step padded to its CUDA graph's size routes its padding rows like any
token, so a few tokens per step land where the padding's stale inputs send them. The
reader says so; a heatmap read at a glance is not changed by it.

Off the hot path, a daemon thread writes the pinned matrix to the file twice a second
(wire.py). It touches CPU memory and the file alone: a CUDA call from a second thread while
the main thread captures a graph would end the capture, and the boot with it.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

from . import wire

logger = logging.getLogger("flash_local_experts")

DEFAULT_PATH = "/dev/shm/qwen38-experts.bin"
PERIOD_S = 0.5


def file_path() -> str:
    return os.environ.get("QWEN38_EXPERTS_FILE") or DEFAULT_PATH


class Counter:
    """The matrix and its writer. counts and host are [capacity, experts] int32 tensors on the
    device and in pinned host memory; ones is a device int32 vector of ones, at least as long
    as any step's tokens x top_k. Built by build() in the engine, by a test with its fakes."""

    def __init__(self, counts, ones, host, experts: int, top_k: int, meta: dict, path: str = ""):
        self.counts, self.ones, self.host = counts, ones, host
        self.experts, self.top_k = experts, top_k
        self.capacity = int(counts.shape[0])
        self.meta = dict(meta)
        self.names: list[str] = []
        self.path = path or file_path()
        self.seq = 0
        self.pid = os.getpid()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._said = False

    def add_row(self, name: str) -> int | None:
        """The row for a runner, in construction order; None once the matrix is full (a model
        with more MoE layers than the config said: counted up to capacity, the rest not)."""
        with self._lock:
            if len(self.names) >= self.capacity:
                logger.warning("flash_local_experts: no row left for %s (capacity %d)", name, self.capacity)
                return None
            self.names.append(name)
            return len(self.names) - 1

    def capture_fn(self, row: int, previous: Callable | None = None) -> Callable:
        """The slot's function for one row. The first row's call also queues the copy out."""
        counts_row = self.counts[row]
        counts, ones, host = self.counts, self.ones, self.host
        width = ones.numel()
        first = row == 0

        def capture(topk_ids):
            ids = topk_ids.reshape(-1)
            n = ids.numel()
            if n <= width:
                counts_row.index_add_(0, ids, ones[:n])
            if first:
                host.copy_(counts, non_blocking=True)
            if previous is not None:
                previous(topk_ids)

        return capture

    # ── the writer ──────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._loop, name="flash-local-experts", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            time.sleep(PERIOD_S)
            try:
                self.write_once()
            except Exception as e:  # noqa: BLE001 (a writer that dies takes the view with it, nothing else)
                if not self._said:
                    self._said = True
                    logger.warning("flash_local_experts: the counter file is not written: %s: %s",
                                   type(e).__name__, e)

    def frame(self) -> bytes:
        """The file's bytes as of now: the rows registered so far, from the host copy."""
        with self._lock:
            names = list(self.names)
        rows = len(names)
        self.seq += 1
        meta = dict(self.meta, rows=names)
        counts = self.host[:rows].numpy().tobytes() if rows else b""
        return wire.pack(meta, counts, rows, self.experts, self.top_k, self.seq, time.time(), self.pid)

    def write_once(self) -> None:
        data = self.frame()
        tmp = f"{self.path}.{self.pid}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)   # the cockpit reads it as the operator, the engine writes it as root
        os.replace(tmp, self.path)


def model_meta(cfg) -> dict:
    """The model's shape for the drawing, read off the engine's config with every field
    optional: a checkpoint without one of them draws without it."""
    hf = getattr(cfg.model_config, "hf_text_config", None) or getattr(cfg.model_config, "hf_config", None)
    get = lambda k, d=None: getattr(hf, k, d) if hf is not None else d  # noqa: E731
    spec = getattr(cfg, "speculative_config", None)
    served = getattr(cfg.model_config, "served_model_name", None)
    if isinstance(served, (list, tuple)):
        served = served[0] if served else None
    meta = {
        "model": getattr(cfg.model_config, "model", None),
        "served_model_name": served,
        "model_type": get("model_type"),
        "num_hidden_layers": get("num_hidden_layers"),
        "layer_types": get("layer_types"),
        "full_attention_interval": get("full_attention_interval"),
        "hidden_size": get("hidden_size"),
        "num_attention_heads": get("num_attention_heads"),
        "num_key_value_heads": get("num_key_value_heads"),
        "num_experts": get("num_experts"),
        "num_experts_per_tok": get("num_experts_per_tok"),
        "moe_intermediate_size": get("moe_intermediate_size"),
        "shared_expert_intermediate_size": get("shared_expert_intermediate_size"),
        "hc_count": get("hc_count"),
        "ple_layer_ids": get("ple_layer_ids"),
        "mtp_num_hidden_layers": get("mtp_num_hidden_layers"),
        "vocab_size": get("vocab_size"),
        "spec_method": getattr(spec, "method", None) if spec is not None else None,
        "num_speculative_tokens": getattr(spec, "num_speculative_tokens", None) if spec is not None else None,
        "draft_model": getattr(getattr(spec, "draft_model_config", None), "model", None) if spec is not None else None,
        "max_num_seqs": getattr(cfg.scheduler_config, "max_num_seqs", None),
        "max_num_batched_tokens": getattr(cfg.scheduler_config, "max_num_batched_tokens", None),
    }
    out = {}
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            v = [x for x in v if isinstance(x, (int, float, str))]
        if v is None or isinstance(v, (int, float, str, list)):
            out[k] = v
    return out


def build(cfg, experts: int, top_k: int) -> Counter:
    """The counter for this engine, on the model's device, sized by its config."""
    import torch

    hf = getattr(cfg.model_config, "hf_text_config", None)
    layers = int(getattr(hf, "num_hidden_layers", 0) or 0)
    extra = int(getattr(hf, "mtp_num_hidden_layers", 0) or 0)
    capacity = max(layers + extra, 1) + 4
    tokens = int(getattr(cfg.scheduler_config, "max_num_batched_tokens", 0) or 0) or 8192
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    counts = torch.zeros((capacity, experts), dtype=torch.int32, device=device)
    ones = torch.ones(tokens * top_k * 2, dtype=torch.int32, device=device)
    # the host copy, on the CPU by name: the first MoE layer is built inside vLLM's device context,
    # where a tensor with no device lands on the GPU and cannot be pinned (boot of 2026-11-10,
    # "Only dense CPU tensors can be pinned", 48 layers not counted); unpinned when pinning fails,
    # the copy out then waits on the device instead of running beside it
    host = torch.zeros((capacity, experts), dtype=torch.int32, device="cpu")
    if cuda:
        try:
            host = host.pin_memory()
        except RuntimeError as e:
            logger.warning("flash_local_experts: the host copy is not pinned, copies out will wait: %s", e)
    return Counter(counts, ones, host, experts, top_k, model_meta(cfg))
