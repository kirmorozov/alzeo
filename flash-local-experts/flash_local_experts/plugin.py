"""The vLLM general plugin entry for flash-local-experts.

vLLM's plugin loader calls register() in the API server, the engine core and every worker
before a model is built. Only a process that builds a model constructs MoERunners, so only
there does the counter come to be; the others load a wrapper that is never called.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("flash_local_experts")

# Set on the class so installing twice, which a worker's reload does, never wraps the
# constructor around itself.
_MARKER = "_flash_local_experts_counts"

_STATE: dict = {"counter": None}


def register() -> None:
    if os.environ.get("QWEN38_EXPERTS", "1") == "0":
        return
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    install(MoERunner)


def install(runner_cls, make=None) -> None:
    """Wrap the runner's constructor: once the runner stands, its router gets the counter's
    function for the runner's row. A failure here is logged and the model is built as it
    would be without the plugin: a view is never worth a boot."""
    if getattr(runner_cls, _MARKER, False):
        return
    original = runner_cls.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            attach(self, make or _build)
        except Exception as e:  # noqa: BLE001 (isolation by design)
            logger.warning("flash_local_experts: %s is not counted: %s: %s",
                           getattr(self, "layer_name", "?"), type(e).__name__, e)

    runner_cls.__init__ = __init__
    setattr(runner_cls, _MARKER, True)


def _build(router):
    from vllm.config import get_current_vllm_config

    from .counter import build

    return build(get_current_vllm_config(), int(router.global_num_experts), int(router.top_k))


def attach(runner, make) -> bool:
    """The runner's router gets the counter's function for a new row. False when the router
    has no slot (another router class), or when the matrix is full."""
    router = getattr(runner, "router", None)
    if router is None or not callable(getattr(router, "set_capture_fn", None)):
        return False
    counter = _STATE["counter"]
    if counter is None:
        counter = _STATE["counter"] = make(router)
        counter.start()
        logger.info("flash_local_experts: counting %d experts, top %d, into %s",
                    counter.experts, counter.top_k, counter.path)
    if int(router.global_num_experts) != counter.experts:
        return False
    row = counter.add_row(str(getattr(runner, "layer_name", "")))
    if row is None:
        return False
    router.set_capture_fn(counter.capture_fn(row, getattr(router, "capture_fn", None)))
    return True
