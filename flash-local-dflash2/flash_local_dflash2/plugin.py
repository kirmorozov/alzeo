"""The vLLM general plugin entry for flash-local-dflash2.

vLLM's plugin loader calls register() in the engine core and in every worker
before a model is built. The arch is registered by string, so the worker
imports flash_local_dflash2.model only when it resolves
Qwen4ExpForConditionalGeneration: importing vllm here, at plugin-load time,
would pull model code into every process that loads plugins, the way the
image's own registry avoids that with its string form.
"""


def register() -> None:
    """Re-register the lane target's arch with the DFlash2 capture subclass, and open the
    image's speculative-method whitelist for it.

    The two belong together: the image's hook for this arch refuses ``dflash`` while the
    engine's config is built, before any model class is resolved, so a lane that mounted
    only the model subclass would die before it was ever asked for.
    """
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import install

    # register_model overrides the image's entry for the arch; the lane serves
    # only this arch, and only on the CUDA build the image ships.
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "flash_local_dflash2.model:DFlashQwen4ExpForConditionalGeneration",
    )
    install(MODELS_CONFIG_MAP)
