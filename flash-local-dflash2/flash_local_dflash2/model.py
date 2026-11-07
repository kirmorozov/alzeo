"""The DFlash2 capture subclass of the lane image's Qwen4Exp target.

The image's Qwen4ExpForConditionalGeneration inherits the EAGLE3 interface,
but Qwen4ExpModel.forward never reads the aux layers, and the inherited
set_aux_hidden_state_layers asserts an EagleModelMixin the model does not
carry, so `--speculative-config.method dflash` dies in the model runner
before the first forward. This subclass stores the ids the runner hands
over, wraps the five completed layers they name, and makes the outer forward
return the (hidden_states, aux list) tuple the runner unpacks when it runs
with aux hidden state outputs. The capture writes into one fixed
preallocated buffer, the model's own MTP-capture pattern (_mtp_hidden_buffer),
so the CUDA graphs the b12x runner captures over the eager forward replay the
same addresses. The target is not torch-compiled in this image (no
support_torch_compile on the nvidia model), so there is no graph signature to
keep; the separate compile cache the lane gives this target guards the b12x
and FlashInfer kernel choices instead.
"""
from __future__ import annotations

import torch

from vllm.distributed import get_pp_group
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
from vllm.sequence import IntermediateTensors

from .capture import DFlashCaptureLayer


class DFlashQwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    """Qwen4Exp target that serves as the DFlash2 target.

    Only the EAGLE3 hooks and the outer forward's return change. The layer
    loop runs the image's code through the wrapped layers, and the capture
    lands in one fixed buffer, so nothing the CUDA graphs rely on moves.
    """

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Set once by set_aux_hidden_state_layers, read in forward: the
        # forward's return shape is a static choice, not a branch on data.
        self._dflash_capture_slots = 0

    def set_aux_hidden_state_layers(self, layer_ids: tuple[int, ...]) -> None:
        # The runner hands the drafter's target_layer_ids plus one: the input
        # of layer i + 1 is the completed output of layer i, and that is what
        # the drafter was trained on.
        capture_ids = tuple(i - 1 for i in layer_ids)
        if capture_ids != tuple(sorted(set(capture_ids))):
            raise ValueError(
                f"capture layers must be distinct and ascending: {layer_ids}"
            )
        language_model = self.language_model
        inner = language_model.model
        tp_size = language_model.vllm_config.parallel_config.tensor_parallel_size
        # The reference capture is validated for TP1/CP1. HC
        # sequence-parallel needs no check of its own: the layers refuse it
        # at construction, so a model that reached this point cannot shard
        # the multi-stream state.
        assert tp_size == 1, f"DFlash2 capture is validated for TP1, found {tp_size}"
        assert get_pp_group().world_size == 1, (
            "DFlash2 capture is validated without pipeline parallelism"
        )
        if self._dflash_capture_slots:
            raise RuntimeError("aux layers set twice")
        if not capture_ids or not all(
            inner.start_layer <= i < inner.end_layer for i in capture_ids
        ):
            raise ValueError(
                f"capture layers {capture_ids} fall outside "
                f"layers {inner.start_layer}..{inner.end_layer - 1}"
            )
        hc_count = inner.config.hc_count
        hidden_size = inner.config.hidden_size
        buffer = torch.empty(
            len(capture_ids),
            language_model.scheduler_config.max_num_batched_tokens,
            hidden_size,
            dtype=language_model.model_config.dtype,
            device=inner.embed_tokens.weight.device,
        )
        inner.register_buffer("_dflash_aux_buffer", buffer, persistent=False)
        for slot, layer_idx in enumerate(capture_ids):
            inner.layers[layer_idx] = DFlashCaptureLayer(
                inner.layers[layer_idx], buffer, slot, hc_count, hidden_size
            )
        self._dflash_capture_slots = len(capture_ids)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # The drafter's config carries the ids on this lane, so this default
        # only says what the target captures when no drafter config is
        # present.
        return (6, 16, 26, 36, 46)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # The image's outer forward as it is (deepstack staging, the n-gram
        # kwargs), plus the aux return the runner unpacks when it runs with
        # aux hidden state outputs. Not a copy of that forward: a copy would
        # drift from the image's.
        hidden_states = super().forward(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if not self._dflash_capture_slots or isinstance(
            hidden_states, IntermediateTensors
        ):
            return hidden_states
        buffer = self.language_model.model._dflash_aux_buffer
        num_tokens = hidden_states.shape[0]
        aux_hidden_states = [
            buffer[slot, :num_tokens] for slot in range(self._dflash_capture_slots)
        ]
        return hidden_states, aux_hidden_states
