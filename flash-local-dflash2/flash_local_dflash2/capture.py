"""The capture itself, torch only: what the test runs without the image's vLLM.

DFlashCaptureLayer wraps one of the target's decoder layers and writes the
DFlash2 feature of its completed state into a fixed buffer; capture_hc_mean is
the arithmetic. model.py, which needs vLLM, binds them to the target.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class DFlashCaptureLayer(nn.Module):
    """A Qwen4ExpDecoderLayer plus the DFlash2 capture of its completed state.

    The target defers its hyper-connection combine between layers, so the
    completed output of this layer (the state the next one starts from)
    exists only as a pending triple until the next layer's first step
    materializes it. Materializing it here costs one combine and one mean
    per point and adds no state of its own, provided the combine is a pure
    function of the triple: the triple goes on to the next layer as this
    layer returned it, so nothing the combine writes may alias it. The image's
    deepstack and PP paths call the layer's own ``mlp_hyper_connection.combine``
    at this very point, but they drop the triple afterwards, and under b12x
    that method runs through the model's shared hyper-connection workspace,
    whose output storage this wrapper cannot vouch for. ``combine`` therefore
    names the function to use; the model hands the image's plain
    ``hc_combine`` custom op, which allocates its output (``residual.new_empty``
    in the image's ``qwen4_exp/amd/ops/hc.py``) and so can alias nothing. Left
    None, the layer's own method is used, which is what the torch-only test
    exercises.

    The wrapper stands in the model's layer list, so everything that reads a
    layer's attributes after the swap (the forward loop's deepstack combine,
    the PP tail, the MoE and state walks) must still see the decoder layer:
    attribute lookups that the wrapper itself cannot answer go to it.
    """

    def __init__(
        self,
        layer: nn.Module,
        buffer: torch.Tensor,
        slot: int,
        hc_count: int,
        hidden_size: int,
        combine: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        # Not an attribute nn.Module would register: a plain callable, read in
        # forward; None means the wrapped layer's own combine (see the class).
        self._combine = combine
        # A plain attribute, not a registered buffer: the model owns the one
        # buffer, and a second registration would put it in every wrapper's
        # state.
        self._buffer = buffer
        self._slot = slot
        self._hc_count = hc_count
        self._hidden_size = hidden_size

    def __getattr__(self, name: str):
        # nn.Module resolves parameters, buffers and submodules first; what is
        # left (layer_type, layer_idx, self_attn, mlp_hyper_connection, ...)
        # belongs to the wrapped layer.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "layer":
                raise
            return getattr(self.layer, name)

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
        combine = self._combine
        if combine is None:
            combine = self.layer.mlp_hyper_connection.combine
        capture_hc_mean(
            combine(hidden_states, block_output, injection),
            self._buffer,
            self._slot,
            self._hc_count,
            self._hidden_size,
        )
        return hidden_states, block_output, injection


def capture_hc_mean(
    combined: torch.Tensor,
    buffer: torch.Tensor,
    slot: int,
    hc_count: int,
    hidden_size: int,
) -> None:
    """Average the hyper-connection streams of a materialized [tokens, hc*H]
    state and copy the rows into the buffer's slot, in place.

    DFlash2 was trained on the completed, zero-based layer outputs, averaged
    over the four hyper-connection streams (the drafter's own SGLang capture
    patch). The rows land at a fixed address, so a CUDA graph replayed at the
    same token count overwrites the same bytes.
    """
    assert combined.shape[-1] == hc_count * hidden_size, combined.shape
    num_tokens = combined.shape[0]
    assert num_tokens <= buffer.shape[1], (num_tokens, buffer.shape)
    buffer[slot, :num_tokens].copy_(
        combined.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)
    )
