"""Read-only diagnostics for the DFlash2 KV-cache layout failure (Failure B).

The dflash boot dies in ``vllm.v1.kv_cache_interface.create_kv_cache_views`` with a
message that names two byte strides (``block stride 30801920 != page 6160384``) and a
split (``47 kernel blocks of 64 tokens``) but not the specs that produced them, and the
image is pinned so nothing can be added inside it. These wrappers call straight through
and record what the builder was actually handed: the KV cache spec of each group, the
kernel block size chosen for it, and the tensor's own strides.

Nothing here changes what vLLM does - every path returns the callee's result or re-raises
unchanged. Mounted through the plugin, so the image is never edited. Delete once Failure
B is fixed.
"""

from __future__ import annotations

import sys

from vllm.logger import init_logger

_logger = init_logger(__name__)

_FIELDS = (
    "block_size",
    "num_heads",
    "num_kv_heads",
    "head_size",
    "head_size_v",
    "dtype",
    "storage_block_size",
    "page_size_padded",
    "page_tail_bytes_per_token",
    "tokens_per_state",
    "sliding_window",
    "page_size_bytes",
)


def _describe(spec: object) -> str:
    """The spec's own geometry, whatever kind of spec it is."""
    bits = []
    for name in _FIELDS:
        if hasattr(spec, name):
            bits.append(f"{name}={getattr(spec, name)}")
    for bs in (16, 64):
        try:
            bits.append(f"kernel_states@{bs}={spec.get_num_kernel_states(bs)}")  # type: ignore[attr-defined]
        except Exception:
            pass
    return f"{type(spec).__name__}({', '.join(bits)})"


def _describe_tensor(tensor: object) -> str:
    return (
        f"layers={len(getattr(tensor, 'layers', []) or [])} "
        f"layer_stride={getattr(tensor, 'layer_stride', '?')} "
        f"block_stride={getattr(tensor, 'block_stride', '?')}"
    )


def _wrap_create(original):
    def create_kv_cache_views(raw, spec, num_blocks, layout, kv_cache_tensor, kernel_block_size=None):
        layout_name = getattr(layout, "name", layout)
        try:
            views = original(
                raw, spec, num_blocks, layout, kv_cache_tensor, kernel_block_size=kernel_block_size
            )
        except Exception:
            _logger.error(
                "df2 kvdiag: create_kv_cache_views RAISED layout=%s kernel_block_size=%s "
                "num_blocks=%s tensor(%s) spec=%s",
                layout_name,
                kernel_block_size,
                num_blocks,
                _describe_tensor(kv_cache_tensor),
                _describe(spec),
            )
            raise
        _logger.info(
            "df2 kvdiag: create_kv_cache_views ok layout=%s kernel_block_size=%s "
            "num_blocks=%s views=%d spec=%s",
            layout_name,
            kernel_block_size,
            num_blocks,
            len(views),
            _describe(spec),
        )
        return views

    return create_kv_cache_views


def _wrap_prepare(original):
    def prepare_kernel_block_sizes(kv_cache_config, attn_groups):
        sizes = original(kv_cache_config, attn_groups)
        _logger.info("df2 kvdiag: kernel block sizes per group = %s", sizes)
        for i, group in enumerate(getattr(kv_cache_config, "kv_cache_groups", []) or []):
            spec = group.kv_cache_spec
            if hasattr(spec, "kv_cache_specs"):
                spec = next(iter(spec.kv_cache_specs.values()), spec)
            _logger.info(
                "df2 kvdiag: group %d layers=%d kernel_block_size=%s spec=%s",
                i,
                len(group.layer_names),
                sizes[i] if i < len(sizes) else None,
                _describe(spec),
            )
        return sizes

    return prepare_kernel_block_sizes


_installed = False

# The helper and its call sites; both callers do ``from ... import create_kv_cache_views``
# at module scope, so every holder has to be repointed, not just the defining module.
_TARGETS = (
    ("vllm.v1.kv_cache_interface", "create_kv_cache_views"),
    ("vllm.v1.worker.utils", "create_kv_cache_views"),
    ("vllm.v1.worker.utils", "prepare_kernel_block_sizes"),
    ("vllm.v1.hisparse.binding", "create_kv_cache_views"),
)


def install() -> None:
    """Wrap the two KV-layout helpers, wherever they are already bound.

    A missing module or attribute is skipped: this reports on a boot, it must never be the
    reason one fails.
    """
    global _installed
    if _installed:
        return
    _installed = True

    modules: dict[str, object] = {}
    for module_name, _attr in _TARGETS:
        if module_name not in modules:
            try:
                modules[module_name] = __import__(module_name, fromlist=["_"])
            except Exception:
                _logger.warning("df2 kvdiag: %s is not importable, skipping it", module_name)
                modules[module_name] = None  # type: ignore[assignment]

    originals: dict[str, object] = {}
    for module_name, attr in _TARGETS:
        original = getattr(modules.get(module_name), attr, None)
        if original is not None:
            originals.setdefault(attr, original)

    if "create_kv_cache_views" not in originals or "prepare_kernel_block_sizes" not in originals:
        _logger.warning("df2 kvdiag: vLLM exposes no %s; diagnostics off", " / ".join(_TARGETS))
        return

    wrappers = {
        "create_kv_cache_views": _wrap_create(originals["create_kv_cache_views"]),
        "prepare_kernel_block_sizes": _wrap_prepare(originals["prepare_kernel_block_sizes"]),
    }
    for attr, wrapper in wrappers.items():
        for module in list(sys.modules.values()):
            if getattr(module, attr, None) is originals[attr]:
                setattr(module, attr, wrapper)
    _logger.info("df2 kvdiag: watching %s", ", ".join(sorted(wrappers)))
