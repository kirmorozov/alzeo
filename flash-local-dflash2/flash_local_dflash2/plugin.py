"""The vLLM general plugin entry for flash-local-dflash2.

vLLM's plugin loader calls register() in the engine core and in every worker
before a model is built. The arch is registered by string, so the worker
imports flash_local_dflash2.model only when it resolves
Qwen4ExpForConditionalGeneration: importing vllm here, at plugin-load time,
would pull model code into every process that loads plugins, the way the
image's own registry avoids that with its string form.
"""


def register() -> None:
    """Re-register the lane target's arch with the DFlash2 capture subclass."""
    from vllm.model_executor.models import ModelRegistry

    # register_model overrides the image's entry for the arch; the lane serves
    # only this arch, and only on the CUDA build the image ships.
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "flash_local_dflash2.model:DFlashQwen4ExpForConditionalGeneration",
    )
