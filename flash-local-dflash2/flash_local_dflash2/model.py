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
from vllm.models.qwen4_exp.nvidia.hyperconnection import hc_combine
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
        # [max_num_batched_tokens, hidden_size], filled in forward. See
        # get_mtp_target_hidden_states for why the drafter needs one.
        self._dflash_last_hidden_buffer = None
        inner = self.language_model.model
        spec_config = vllm_config.speculative_config
        if (
            spec_config is not None
            and getattr(spec_config, "method", None) == "dflash"
        ):
            self._dflash_last_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                inner.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=inner.embed_tokens.weight.device,
            )

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

        # The image's plain combine op, not the layer's workspace-bound method:
        # the triple the wrapper materializes goes on to the next layer
        # untouched, so the combine must write nothing that aliases it, and
        # this op allocates its output (DFlashCaptureLayer says why).
        def combine(hidden_states, block_output, injection):
            return hc_combine(hidden_states, block_output, injection, hc_count)

        for slot, layer_idx in enumerate(capture_ids):
            inner.layers[layer_idx] = DFlashCaptureLayer(
                inner.layers[layer_idx], buffer, slot, hc_count, hidden_size,
                combine=combine,
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
        last_buffer = self._dflash_last_hidden_buffer
        if last_buffer is not None:
            # Same address every step, so the CUDA graphs the b12x runner
            # captures over this forward replay a write that is really there.
            last_buffer[:num_tokens].copy_(hidden_states)
        aux_hidden_states = [
            buffer[slot, :num_tokens] for slot in range(self._dflash_capture_slots)
        ]
        return hidden_states, aux_hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor:
        """The target's final hidden states, for the runner's spec-dec feed.

        The runner feeds the drafter ``last_hidden_states`` and, when the
        target answers to ``get_mtp_target_hidden_states``, it overrides that
        with what this returns, sliced to the active token count
        (``v1/worker/gpu/model_runner.py``: ``pre_hc_hidden_states[:n]``, at
        both the profiling ``_dummy_run`` and the real propose path). It makes
        no None check. The image's Qwen4Exp does answer to that name - MTP uses
        it for the pre-final-mixer multi-stream state ``[T, hc_count * H]``,
        which the image allocates only under ``method == "mtp"``. So under
        ``method == "dflash"`` the inherited method returns None and the slice
        raises ``TypeError: 'NoneType' object is not subscriptable`` in
        ``determine_available_memory``, before a single token is drafted.

        Overriding it here, rather than deleting it, is the only option: the
        method is inherited, so ``hasattr`` is true whatever this class does.
        Returning the ordinary post-mixer hidden states is what the generic
        path would have passed anyway (``spec_hidden_states = hidden_states``).
        The DFlash2 drafter does not want the multi-stream state either: its
        ``fc`` is built at ``target_hidden_size * len(aux_layers)`` (5 x 2560 =
        12800, the width of ``fc.weight`` in the drafter), so its features come
        from the aux list and this argument stays a plain ``[T, H]``.
        """
        buffer = self._dflash_last_hidden_buffer
        if buffer is None:
            raise RuntimeError(
                "the DFlash2 target was built without its last-hidden capture; "
                "this target is only meant for --speculative-config.method dflash"
            )
        return buffer
