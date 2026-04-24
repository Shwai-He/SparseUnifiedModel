# SPDX-License-Identifier: Apache-2.0
"""LayerCompressionMixin — shared record/skip instrumentation for all decoder layers.

Usage (multiple inheritance, mixin must come first):

    from modeling.compression_mixin import LayerCompressionMixin

    class MyDecoderLayer(LayerCompressionMixin, nn.Module):
        def __init__(self, config, layer_idx, record=False):
            super().__init__()          # initialises nn.Module
            self._init_compression(record)
            ...

        def forward(self, hidden_states, ...):
            residual = hidden_states
            self._record_input(residual)          # before attention

            # ... run attention ...
            if not self._should_skip_attn():
                hidden_states = residual + hidden_states
            else:
                hidden_states = residual

            residual = hidden_states
            self._record_residual(residual)       # after attn+residual, before MLP

            if not self._should_skip_mlp():
                # ... run MLP + residual ...

            self._record_output(hidden_states)    # after MLP skip decision

BAGEL note: pass `mode="und"/"gen"` to all three _record_* and _should_skip_* helpers
so that recording is gated on the currently active modality.
"""

import torch
import torch.nn as nn


class LayerCompressionMixin:
    """Adds calibration-record and layer-drop-skip capabilities to any decoder layer.

    Attributes set by _init_compression():
        record (bool):      enable activation accumulation during calibration.
        skip_mode (str|None): for Ming/Qwen — "block"/"attn"/"mlp" or None.
                             for BAGEL  — "und"/"gen" (which modality to gate).
        skip_type (str|None): BAGEL only — "block"/"attn"/"mlp". Leave None for
                             Ming/Qwen (they encode skip intent in skip_mode directly).
        input / residual / output (Tensor|None): accumulated activations.
    """

    def _init_compression(self, record: bool = False) -> None:
        self.record = record
        self.skip_mode = None
        self.skip_type = None   # BAGEL: what to drop; None for Ming/Qwen
        self.input: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None
        self.output: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    #  Internal helper                                                     #
    # ------------------------------------------------------------------ #

    def _accumulate(self, attr: str, tensor: torch.Tensor) -> None:
        # Packed sequences (BAGEL) are 2-D; batched tensors (Ming/Qwen) are 3-D (1,S,H).
        t = tensor.squeeze(0) if tensor.dim() > 2 else tensor
        existing = getattr(self, attr)
        setattr(self, attr, torch.cat((existing, t), 0) if existing is not None else t)

    # ------------------------------------------------------------------ #
    #  Record helpers — call at the right points in forward()             #
    # ------------------------------------------------------------------ #

    def _record_input(self, tensor: torch.Tensor, mode: str = None) -> None:
        """Call with the layer input (pre-attention residual)."""
        if self.record and (mode is None or mode == self.skip_mode):
            self._accumulate("input", tensor)

    def _record_residual(self, tensor: torch.Tensor, mode: str = None) -> None:
        """Call with the post-attention+residual state (pre-MLP residual)."""
        if self.record and (mode is None or mode == self.skip_mode):
            self._accumulate("residual", tensor)

    def _record_output(self, tensor: torch.Tensor, mode: str = None) -> None:
        """Call with the final output (post-MLP, after skip decision for Ming/Qwen;
        pre-skip for BAGEL to capture the raw MLP delta)."""
        if self.record and (mode is None or mode == self.skip_mode):
            self._accumulate("output", tensor)

    # ------------------------------------------------------------------ #
    #  Skip helpers — use in forward() instead of inline if-chains        #
    # ------------------------------------------------------------------ #

    def _should_skip_attn(self, mode: str = None) -> bool:
        """Return True when the attention block should be skipped."""
        if self.skip_type is not None:
            # BAGEL: skip_type says what to drop, skip_mode says which modality
            return self.skip_type in ("block", "attn") and self.skip_mode == mode
        # Ming / Qwen: skip_mode directly encodes what to drop
        return self.skip_mode in ("block", "attn")

    def _should_skip_mlp(self, mode: str = None) -> bool:
        """Return True when the MLP block should be skipped."""
        if self.skip_type is not None:
            return self.skip_type in ("block", "mlp") and self.skip_mode == mode
        return self.skip_mode in ("block", "mlp")
