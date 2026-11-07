"""The DFlash2 capture subclass of the lane image's Qwen4Exp target.

The image's Qwen4ExpForConditionalGeneration inherits the EAGLE3 interface,
but Qwen4ExpModel.forward never reads the aux layers, and the inherited
set_aux_hidden_state_layers asserts an EagleModelMixin the model does not
carry, so `--speculative-config method:dflash` dies in the model runner
before the first forward. This subclass stores the ids the runner hands
over, wraps the five completed layers they name, and makes the outer forward
return the (hidden_states, aux list) tuple the runner unpacks when it runs
with aux hidden state outputs. The capture writes into one fixed
preallocated buffer, the model's own MTP-capture pattern, so torch.compile
and CUDA graphs hold.
"""
from __future__ import annotations

import torch
from torch import nn

from vllm.distributed import get_pp_group
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration
from vllm.sequence import IntermediateTensors


class DFlashCaptureLayer(nn.Module):
    """A Qwen4ExpDecoderLayer plus the DFlash2 capture of its completed state.

    The target defers its hyper-connection combine between layers, so the
    completed output of this layer (the state the next one starts from)
    exists only as a pending triple until the next layer's first step
    materializes it. Materializing it here calls the same pure combine the
    deepstack and PP paths already use: it reads the triple and this layer's
    combine plan, writes a new tensor, and changes nothing, so capturing
    costs one combine and one mean per point and adds no state of its own.
    """

    def __init__(
        self,
        layer: nn.Module,
        buffer: torch.Tensor,
        slot: int,
        hc_count: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self._buffer = buffer
        self._slot = slot
        self._hc_count = hc_count
        self._hidden_size = hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
        output_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, block_output, injection = self.layer(
            hidden_states=hidden_states,
            prev_block_output=prev_block_output,
            prev_injection=prev_injection,
            positions=positions,
            input_ids=input_ids,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
            output_indices=output_indices,
        )
        # DFlash2 was trained on the completed, zero-based layer outputs,
        # averaged over the four hyper-connection streams.
        combined = self.layer.mlp_hyper_connection.combine(
            hidden_states, block_output, injection
        )
        assert combined.shape[-1] == self._hc_count * self._hidden_size
        view = self._buffer[self._slot, : combined.shape[0]]
        view.copy_(
            combined.unflatten(-1, (self._hc_count, self._hidden_size)).mean(
                dim=-2
            )
        )
        return hidden_states, block_output, injection


class DFlashQwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    """Qwen4Exp target that serves as the DFlash2 target.

    Only the EAGLE3 hooks and the outer forward change. The layer loop runs
    the image's code through the wrapped layers, and the capture lands in one
    fixed buffer, so nothing the compiled forward or its CUDA graphs rely on
    moves.
    """

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Set once by set_aux_hidden_state_layers, read in forward: the
        # compiled forward's return shape is a static choice, not a branch
        # on data.
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
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # The image's outer forward, kept as is, plus the aux return the
        # runner unpacks when it runs with aux hidden state outputs.
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        if self._dflash_capture_slots:
            buffer = self.language_model.model._dflash_aux_buffer
            num_tokens = hidden_states.shape[0]
            aux_hidden_states = [
                buffer[slot, :num_tokens]
                for slot in range(self._dflash_capture_slots)
            ]
            return hidden_states, aux_hidden_states
        return hidden_states
