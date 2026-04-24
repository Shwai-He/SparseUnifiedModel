# SPDX-License-Identifier: Apache-2.0
import os
import sys
import time
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .base import BaseGenModel

# diffusers is vendored under modeling/; add it to path so `import diffusers` resolves.
_MODELING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "modeling")
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

        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        lm_path = f"{model_path}/text_encoder"

        # Unconditionally replace text encoder with our instrumented version.
        del pipe.text_encoder
        torch.cuda.empty_cache()
        pipe.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            lm_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        ).to(torch.bfloat16)
        return cls(pipe, device)

    def generate(self, prompt: str, num_images: int = 1, cfg_scale: float = 4.0,
                 num_timesteps: int = 50, resolution: int = 1024,
                 device: str = "cuda", **kwargs) -> List[Image.Image]:
        # Qwen-Image uses a fixed 16:9 resolution; the resolution arg is unused.
        images = []
        for _ in range(num_images):
            img = self.pipe(
                prompt=prompt,
                negative_prompt=" ",
                width=1664, height=928,
                num_inference_steps=num_timesteps,
                true_cfg_scale=cfg_scale,
                generator=torch.Generator(device="cuda").manual_seed(int(time.time())),
            ).images[0]
            images.append(img)
        return images

    def get_transformer_layers(self) -> List[nn.Module]:
        return self.pipe.text_encoder.model.language_model.layers
