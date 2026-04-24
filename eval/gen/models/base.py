# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image


class BaseGenModel(ABC):
    """Unified interface for image-generation models (BAGEL, Ming, Qwen-Image).

    Subclasses implement load / generate / get_transformer_layers.
    Default weight-pruning hooks cover the Ming / Qwen cases; BAGEL overrides them.
    """

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, device: str, **kwargs) -> "BaseGenModel":
        """Instantiate and return a ready-to-use model."""

    @abstractmethod
    def generate(self, prompt: str, num_images: int = 1, cfg_scale: float = 4.0,
                 num_timesteps: int = 30, resolution: int = 1024,
                 device: str = "cuda", **kwargs) -> List[Image.Image]:
        """Generate *num_images* images for *prompt*. Always returns a list."""

    @abstractmethod
    def get_transformer_layers(self) -> List[nn.Module]:
        """Return the list of transformer decoder layers."""

    # ------------------------------------------------------------------ #
    #  Weight-pruning hooks (override per model as needed)                 #
    # ------------------------------------------------------------------ #

    def has_dual_path(self) -> bool:
        """BAGEL has separate und / gen MLP weights; others return False."""
        return False

    def get_mlp(self, layer: nn.Module, path: str = "und") -> nn.Module:
        """Return the MLP module for *layer* and *path* ('und' or 'gen')."""
        return layer.mlp

    def get_mlp_units(self, layer: nn.Module, path: str = "und") -> List[nn.Module]:
        """Return atomic MLP units to prune (handles MoE expert lists)."""
        mlp = self.get_mlp(layer, path)
        if hasattr(mlp, "experts"):
            return list(mlp.experts)
        return [mlp]

    def get_attn(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def setup_pruning_calibration(self, target_layer: str = "mlp") -> None:
        """Enable activation accumulation on all layers before calibration runs."""
        for layer in self.get_transformer_layers():
            if target_layer == "mlp":
                for unit in self.get_mlp_units(layer, "und"):
                    unit.sparse_mode = "prune"
            # 'attn' is BAGEL-only; subclass overrides

    def apply_pruning(
        self,
        keep_ratio: float,
        compressed_layers_und: Optional[List[int]],
        compressed_layers_gen: Optional[List[int]] = None,
        target_layer: str = "mlp",
        scores: Optional[dict] = None,
    ) -> None:
        """Prune and reset all target modules after calibration."""
        from compress_utils import prune_mlp
        for i, layer in enumerate(self.get_transformer_layers()):
            score = scores[i] if scores is not None else None
            for unit in self.get_mlp_units(layer, "und"):
                prune_mlp(unit, keep_ratio, compressed_layers_und, score)
                unit.sparse_mode = "dense"
