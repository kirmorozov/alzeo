"""One KV cache group per drafter layer, so the pool's block is not sized by the drafter.

vLLM's hybrid KV cache manager draws every group's blocks from one pool, and a block of
that pool is as large as the widest group needs: the sum of the page sizes of the layers
in it (``_get_kv_cache_bytes_per_block``). The image groups the target's layers one per
group (49 groups of one layer, each page 3,272,704 bytes at the 3008-token block the
mamba page dictates), but keeps the DFlash2 drafter's five sliding-window layers together
in one group (``_partition_parallel_draft_specs`` then ``_get_kv_cache_groups_uniform_spec``).
The drafter stores 8 KV heads of 128 against the target's 2 of 256, so its page is
6,160,384 bytes and the group's five pages make every block of the pool 30,801,920 bytes.
Each of the target's 49 groups then holds 3.27 MB of state in a 30.8 MB block, and one
request at the 262,144-token window is billed 35.77 GiB of a 18.75 GiB pool (the boot of
2026-11-08, ``_check_enough_kv_cache_memory``).

Two changes bring the block back to the target's own page. First, the drafter's layers go
in five groups of one, so the widest group is one drafter page, 6,160,384 bytes (the
speculator supports several draft groups by design: ``draft_kv_cache_group_ids`` and
``_layer_group_idx`` in the image's DFlash speculator, each layer reading the block table
of its own group). Second, the drafter's block shrinks to the largest divisor of the
target's block, a multiple of 16, whose page fits under the target's page: 1504 tokens
here, a page of 3,080,192 bytes, so a block of the pool is the target's 3,272,704 bytes
again and the pool holds what it holds for the MTP target. vLLM allows a block size per
group: the scheduler aligns to the least common multiple of the groups' block sizes
(3008), prefix hashing to their greatest common divisor (1504), every group's block must
divide the first and be a multiple of the second (``resolve_kv_cache_block_sizes``, the
coordinator's asserts), and the worker builds each layer's attention from its group's
spec. Triton attention takes any kernel block that is a multiple of 16.

Installed by the plugin's ``register()``, this wraps ``get_kv_cache_groups`` in
``vllm.v1.core.kv_cache_utils`` where the engine core (``get_kv_cache_configs``), the
model runner and the CUDA-graph planner look it up at call time. The wrapper shrinks the
drafter specs' block, calls through, then splits every group whose layers are all drafter
layers, as the image's own partition names them; a group with a target layer in it is left
exactly as it was. Nothing is changed when the method is not ``dflash``.

``DF2_DRAFT_BLOCK_SIZE`` in the engine's environment is the kill switch for the second
change: unset or ``auto`` computes the block, ``off`` keeps the image's, a number forces
one (it must still divide the target's block and be a multiple of 16, or it is refused
and logged). The split has no switch; without it nothing boots at the full window.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import replace

# A child of vLLM's "vllm" logger (see kvdiag.py): the module imports nothing of vLLM at
# load, so the tests can import it where vLLM is not installed.
_logger = logging.getLogger("vllm.flash_local_dflash2.kvgroups")

_MODULE = "vllm.v1.core.kv_cache_utils"
_installed = False


def split_draft_groups(
    groups: Iterable[object],
    is_draft_layer: Callable[[str], bool],
    make_group: Callable[[list[str], object], object],
    per_layer_spec: Callable[[object, str], object],
) -> tuple[list[object], int]:
    """Return the groups with every all-drafter group of several layers split one per layer.

    ``make_group(layer_names, spec)`` builds a group, ``per_layer_spec(group, name)`` gives
    the spec a layer of the group has (groups of one type may carry a merged spec). The
    second value is the number of groups that were split. Pure, so it is tested on plain
    objects without vLLM.
    """
    out: list[object] = []
    split = 0
    for group in groups:
        names = list(getattr(group, "layer_names"))
        if len(names) > 1 and all(is_draft_layer(name) for name in names):
            for name in names:
                out.append(make_group([name], per_layer_spec(group, name)))
            split += 1
        else:
            out.append(group)
    return out, split


def draft_block_size(
    target_page: int, draft_bytes_per_token: int, target_block: int, align: int = 16
) -> int | None:
    """The largest divisor of ``target_block`` that is a multiple of ``align`` and whose
    drafter page, ``d * draft_bytes_per_token``, is at most ``target_page``.

    None when the drafter's page at ``target_block`` already fits, or when no such divisor
    exists. Pure.
    """
    if target_block * draft_bytes_per_token <= target_page:
        return None
    for d in range(target_block - target_block % align, 0, -align):
        if target_block % d == 0 and d * draft_bytes_per_token <= target_page:
            return d
    return None


def _unpadded_page(spec) -> int:
    return getattr(spec, "unpadded_page_size_bytes", spec.page_size_bytes)


def shrink_draft_blocks(
    kv_cache_spec: dict, draft_names: set[str], forced: int | None = None
) -> tuple[dict, int | None, int | None]:
    """Return the specs with every drafter attention spec at the shrunken block, the block
    chosen (None if unchanged) and the target page it was fitted under.

    Drafter specs are attention specs with a ``block_size``; the target page is the largest
    page any target layer has, padded as the image padded it. ``forced`` overrides the
    computation when it divides the drafter's block and is a multiple of 16.
    """
    target_pages = [
        spec.page_size_bytes for name, spec in kv_cache_spec.items() if name not in draft_names
    ]
    drafts = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name in draft_names and hasattr(spec, "block_size")
    }
    if not target_pages or not drafts:
        return kv_cache_spec, None, None
    target_page = max(target_pages)
    blocks = {spec.block_size for spec in drafts.values()}
    per_token = {_unpadded_page(spec) // spec.block_size for spec in drafts.values()}
    if len(blocks) != 1 or len(per_token) != 1:
        _logger.warning(
            "df2 kvgroups: drafter layers disagree on block (%s) or bytes per token (%s); "
            "their block is kept",
            sorted(blocks),
            sorted(per_token),
        )
        return kv_cache_spec, None, target_page
    (block,) = blocks
    (bytes_per_token,) = per_token
    if forced is not None:
        if forced <= 0 or forced % 16 or block % forced:
            _logger.warning(
                "df2 kvgroups: DF2_DRAFT_BLOCK_SIZE=%d does not divide the drafter's block %d "
                "as a multiple of 16; computing it instead",
                forced,
                block,
            )
            forced = None
    new_block = forced if forced is not None else draft_block_size(target_page, bytes_per_token, block)
    if new_block is None or new_block == block:
        return kv_cache_spec, None, target_page
    out = dict(kv_cache_spec)
    for name, spec in drafts.items():
        out[name] = replace(spec, block_size=new_block)
    return out, new_block, target_page


def _forced_block() -> tuple[bool, int | None]:
    """(shrink at all, forced block) from DF2_DRAFT_BLOCK_SIZE."""
    raw = os.environ.get("DF2_DRAFT_BLOCK_SIZE", "auto").strip().lower()
    if raw in ("", "auto"):
        return True, None
    if raw in ("off", "0", "no", "false"):
        return False, None
    try:
        return True, int(raw)
    except ValueError:
        _logger.warning("df2 kvgroups: DF2_DRAFT_BLOCK_SIZE=%r is not a number; computing", raw)
        return True, None


def _wrap(original, kcu):
    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        speculative = getattr(vllm_config, "speculative_config", None)
        if speculative is None or getattr(speculative, "method", None) != "dflash":
            return original(vllm_config, kv_cache_spec)
        try:
            _target_specs, draft_specs = kcu._partition_parallel_draft_specs(
                vllm_config, kv_cache_spec
            )
        except Exception:
            _logger.exception("df2 kvgroups: could not tell drafter layers apart; groups kept")
            return original(vllm_config, kv_cache_spec)
        if not draft_specs:
            return original(vllm_config, kv_cache_spec)
        draft_names = set(draft_specs)

        shrink, forced = _forced_block()
        if shrink:
            try:
                kv_cache_spec, new_block, target_page = shrink_draft_blocks(
                    kv_cache_spec, draft_names, forced
                )
            except Exception:
                _logger.exception("df2 kvgroups: drafter block not shrunk")
            else:
                if new_block is not None:
                    sample = kv_cache_spec[next(iter(draft_names))]
                    _logger.info(
                        "df2 kvgroups: drafter block %d tokens, page %d bytes under the "
                        "target's %d (%d drafter layers)",
                        new_block,
                        sample.page_size_bytes,
                        target_page,
                        len(draft_names),
                    )

        groups = original(vllm_config, kv_cache_spec)
        new_groups, split = split_draft_groups(
            groups,
            draft_names.__contains__,
            lambda names, spec: kcu.KVCacheGroupSpec(names, spec),
            kcu._get_per_layer_spec,
        )
        if not split:
            return groups
        try:
            before = kcu._get_kv_cache_bytes_per_block(groups)
            after = kcu._get_kv_cache_bytes_per_block(new_groups)
        except Exception:
            before = after = -1
        _logger.info(
            "df2 kvgroups: split %d drafter group(s) into one group per layer: "
            "%d -> %d groups, pool bytes per block %d -> %d",
            split,
            len(groups),
            len(new_groups),
            before,
            after,
        )
        return new_groups

    get_kv_cache_groups._df2_kvgroups = True  # type: ignore[attr-defined]
    return get_kv_cache_groups


def install() -> None:
    """Wrap ``get_kv_cache_groups`` wherever a loaded module holds it. Idempotent.

    A missing module or attribute is logged and skipped: the drafter then boots with the
    image's grouping (and dies at the memory check at the full window), it does not die here.
    """
    global _installed
    if _installed:
        return
    _installed = True

    try:
        kcu = __import__(_MODULE, fromlist=["_"])
    except Exception:
        _logger.warning("df2 kvgroups: %s is not importable, skipping", _MODULE)
        return
    original = getattr(kcu, "get_kv_cache_groups", None)
    needed = ("_partition_parallel_draft_specs", "KVCacheGroupSpec", "_get_per_layer_spec")
    missing = [name for name in needed if not hasattr(kcu, name)]
    if original is None or missing:
        _logger.warning(
            "df2 kvgroups: %s lacks %s; drafter groups kept as the image makes them",
            _MODULE,
            ", ".join(missing or ["get_kv_cache_groups"]),
        )
        return
    if getattr(original, "_df2_kvgroups", False):
        return

    wrapper = _wrap(original, kcu)
    for module in list(sys.modules.values()):
        if getattr(module, "get_kv_cache_groups", None) is original:
            setattr(module, "get_kv_cache_groups", wrapper)
    _logger.info("df2 kvgroups: one KV cache group per drafter layer")
