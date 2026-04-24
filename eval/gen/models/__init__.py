# SPDX-License-Identifier: Apache-2.0
from .base import BaseGenModel
from .bagel import BagelGenModel
from .ming import MingGenModel
from .qwen import QwenGenModel

_REGISTRY = {
    "bagel": BagelGenModel,
    "ming": MingGenModel,
    "qwen": QwenGenModel,
}


def load_model(model_type: str, model_path: str, device: str, **kwargs) -> BaseGenModel:
    """Factory: load a generation model by type name.

    model_type: "bagel" | "ming" | "qwen"
    """
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model_type {model_type!r}. Choose from: {list(_REGISTRY)}")
    return _REGISTRY[model_type].load(model_path, device, **kwargs)


__all__ = ["BaseGenModel", "BagelGenModel", "MingGenModel", "QwenGenModel", "load_model"]
