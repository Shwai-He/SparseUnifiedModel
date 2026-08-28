# SPDX-License-Identifier: Apache-2.0
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .base import BaseGenModel

# diffusers is vendored under modeling/; add it to path so `import diffusers` resolves.
_MODELING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "modeling")
_REPO_ROOT = os.path.dirname(_MODELING_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _MODELING_DIR not in sys.path:
    sys.path.insert(0, _MODELING_DIR)


class QwenGenModel(BaseGenModel):

    def __init__(self, pipe, device: str):
        self.pipe = pipe
        self.device = device

    @classmethod
    def load(cls, model_path: str, device: str, **kwargs) -> "QwenGenModel":
        from diffusers import DiffusionPipeline
        from modeling.qwen.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

        pipeline_kwargs = kwargs.pop("pipeline_kwargs", {})
        text_encoder_kwargs = kwargs.pop("text_encoder_kwargs", {})
        replace_text_encoder = kwargs.pop("replace_text_encoder", True)
        enable_model_cpu_offload = kwargs.pop("enable_model_cpu_offload", False)
        enable_sequential_cpu_offload = kwargs.pop("enable_sequential_cpu_offload", False)
        load_on_cpu = kwargs.pop("load_on_cpu", False)
        if kwargs:
            raise TypeError(f"unsupported Qwen load options: {sorted(kwargs)}")

        pipeline_kwargs.setdefault("torch_dtype", torch.bfloat16)
        if replace_text_encoder:
            pipeline_kwargs.setdefault("text_encoder", None)
        pipe = DiffusionPipeline.from_pretrained(model_path, **pipeline_kwargs)
        if "device_map" not in pipeline_kwargs and not load_on_cpu:
            pipe = pipe.to(device)
        lm_path = f"{model_path}/text_encoder"

        if replace_text_encoder:
            torch.cuda.empty_cache()
            text_encoder_kwargs.setdefault("torch_dtype", "auto")
            text_encoder_kwargs.setdefault("device_map", "auto")
            text_encoder_kwargs.setdefault("trust_remote_code", True)
            pipe.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                lm_path, **text_encoder_kwargs
            )
            if "quantization_config" not in text_encoder_kwargs:
                pipe.text_encoder = pipe.text_encoder.to(torch.bfloat16)
        if enable_sequential_cpu_offload:
            if pipe.hf_device_map is not None:
                pipe.reset_device_map()
            pipe.enable_sequential_cpu_offload(device=device)
        elif enable_model_cpu_offload:
            if pipe.hf_device_map is not None:
                pipe.reset_device_map()
            pipe.enable_model_cpu_offload()
        return cls(pipe, device)

    def generate(self, prompt: str, num_images: int = 1, cfg_scale: float = 4.0,
                 num_timesteps: int = 50, resolution: int = 1024,
                 device: str = "cuda", **kwargs) -> List[Image.Image]:
        seed = int(kwargs.pop("seed", 42))
        width = int(kwargs.pop("width", 1664))
        height = int(kwargs.pop("height", 928))
        negative_prompt = kwargs.pop("negative_prompt", " ")
        images = []
        for image_index in range(num_images):
            img = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width, height=height,
                num_inference_steps=num_timesteps,
                true_cfg_scale=cfg_scale,
                generator=torch.Generator(device=device).manual_seed(seed + image_index),
                **kwargs,
            ).images[0]
            images.append(img)
        return images

    def get_transformer_layers(self) -> List[nn.Module]:
        return self.pipe.text_encoder.model.language_model.layers
