"""The lane's answer to the image's rule that this target drafts only with its own MTP head.

The pinned image's hook for this arch, ``Qwen4ExpForConditionalGenerationConfig.
verify_and_update_config`` in ``vllm/model_executor/models/config.py``, refuses every
speculative method but ``mtp``, ``ngram`` and ``ngram_gpu``. It runs while the engine's
config is built, in the API server's ``create_engine_config``, long before the capture
plugin's model class is constructed, so the dflash2 target died there (proved on the
reference box, 2026-11-07):

    NotImplementedError: Qwen4Exp speculative decoding supports only its native MTP
    checkpoint and linear n-gram proposers

That rule is the image's, about the image's own build, and the lane leaves it exactly where
it is: nothing in the container changes, and the flash-local target goes on served by the
image's hook untouched. This module reaches the engine only as the dflash2 target's own
capture plugin, mounted read-only at /opt/df2-plugin on PYTHONPATH, so no other lane and no
other target of this lane ever loads it.

What the lane adds is the one name that whitelist does not carry. The image does ship the
drafter's whole path: ``DFlashModelTypes`` names the method, ``init_speculator`` chooses
``DFlash2Speculator`` for the drafter's ``DFlash2DraftModel`` arch, and
``vllm/model_executor/models/qwen3_dflash2.py`` holds its layers. The refusal is a policy
the image wrote for itself, not a missing implementation.

The image's hook stays the one that decides: this subclass calls it and keeps every check it
makes, the hybrid-cache contract it delegates to Qwen3_5, ``hc_count``, PLE/QSA against
dual-batch overlap and microbatching, PLE against pipeline parallelism, the mrope strip.
Only its last statement, the method whitelist, is stepped over, and only for the one method
this plugin is built for. Anything else the hook raises still goes out to the operator, and
for any other method the image's own answer is passed on unchanged: a ``dflash`` request is
what opens the gate, never an unknown method finding it open.
"""
from __future__ import annotations

# The one name this lane stitches in. The image calls the whole DFlash family "dflash"
# (config/speculative.py's DFlashModelTypes); what makes a draft DFlash2 rather than DFlash
# is its own architecture, DFlash2DraftModel, which vllm's init_speculator reads.
DFLASH_METHOD = "dflash"

# The arch this lane's target is named as: the string its checkpoint's config.json carries,
# and the image's alias of it. Every other arch in the image's map keeps its own hook.
TARGET_ARCHES = (
    "Qwen4ExpForConditionalGeneration",
    "Qwen3_8FlashNextForConditionalGeneration",
)

# The image's own words for the whitelist. Matching them is what keeps the step-over narrow:
# the two refusals this hook also makes (PLE against pipeline parallelism, PLE/QSA against
# dual-batch overlap) say something else and are never swallowed. Should the image reword
# it, the gate stops opening and the refusal reaches the operator again, which is the way
# this fails safe; its test says so.
_GATE = "speculative decoding supports only"

# Set on the subclass so installing twice, which a worker's reload does, never wraps the
# hook around itself.
_MARKER = "_dflash2_opens_the_method_gate"


def opens_the_method_gate(exc: BaseException) -> bool:
    """True for the image's refusal of a drafted target, and for nothing else."""
    return isinstance(exc, NotImplementedError) and _GATE in str(exc)


def _hook_with_the_gate_opened(base: type, method: str) -> type:
    """The image's hook for one arch, with the method whitelist stepped over for ``method``."""

    class DFlash2TargetConfig(base):  # noqa: N801 (named for what it is, on the image's class)
        _dflash2_opens_the_method_gate = True

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            spec = getattr(vllm_config, "speculative_config", None)
            if spec is not None and getattr(spec, "method", None) == method:
                try:
                    return base.verify_and_update_config(vllm_config)
                except NotImplementedError as exc:
                    if not opens_the_method_gate(exc):
                        raise
                    return None
            # Not this lane's drafter: the image's answer, its whitelist included.
            return base.verify_and_update_config(vllm_config)

    DFlash2TargetConfig.__name__ = f"DFlash2{base.__name__}"
    DFlash2TargetConfig.__qualname__ = DFlash2TargetConfig.__name__
    return DFlash2TargetConfig


def install(config_map: dict, arches: tuple[str, ...] = TARGET_ARCHES,
            method: str = DFLASH_METHOD) -> list[str]:
    """Re-point the image's hook for the lane's target arches; say which ones changed.

    Called from the plugin's ``register()``, which vLLM runs in the engine core and in
    every worker before a model is built, and in the API server from ``EngineArgs`` before
    ``create_engine_config`` (vllm/engine/arg_utils.py calls ``load_general_plugins``
    itself) -- so the hook is the lane's by the time the image would have refused.

    The dict is mutated in place on purpose: ``vllm/config/vllm.py`` imports
    ``MODELS_CONFIG_MAP`` by name at the moment it verifies, so a map bound elsewhere here
    would be read as the image's own.
    """
    changed = []
    for arch in arches:
        base = config_map.get(arch)
        if base is None or getattr(base, _MARKER, False):
            # an image without this arch's hook, or a worker that registered twice: nothing
            # to open a second time, and no arch of the image's own is dropped over it
            continue
        config_map[arch] = _hook_with_the_gate_opened(base, method)
        changed.append(arch)
    return changed
