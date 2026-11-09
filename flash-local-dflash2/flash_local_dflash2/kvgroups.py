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

With the drafter's layers in five groups of one, the widest group is one drafter page,
6,160,384 bytes, and the target's groups waste 47 % instead of 89 %: the same request
costs about 7 GiB and the window boots. The speculator supports several draft groups by
design (``draft_kv_cache_group_ids`` and ``_layer_group_idx`` in the image's DFlash
speculator), each layer reading the block table of its own group.

Installed by the plugin's ``register()``, this wraps ``get_kv_cache_groups`` in
``vllm.v1.core.kv_cache_utils`` where the engine core (``get_kv_cache_configs``) and the
CUDA-graph planner look it up at call time. The wrapper calls through, then splits every
group whose layers are all drafter layers, as the image's own partition names them; a
group with a target layer in it is left exactly as it was. Nothing is changed when the
method is not ``dflash``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable

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


def _wrap(original, kcu):
    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        groups = original(vllm_config, kv_cache_spec)
        speculative = getattr(vllm_config, "speculative_config", None)
        if speculative is None or getattr(speculative, "method", None) != "dflash":
            return groups
        try:
            _target_specs, draft_specs = kcu._partition_parallel_draft_specs(
                vllm_config, kv_cache_spec
            )
        except Exception:
            _logger.exception("df2 kvgroups: could not tell drafter layers apart; groups kept")
            return groups
        if not draft_specs:
            return groups
        draft_names = set(draft_specs)
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
