# SPDX-License-Identifier: Apache-2.0
import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .base import BaseGenModel


class MingGenModel(BaseGenModel):

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def load(cls, model_path: str, device: str, **kwargs) -> "MingGenModel":
        from transformers import AutoProcessor
        from modeling.ming.modeling_bailingmm import BailingMMNativeForConditionalGeneration

        model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_image_gen=True,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return cls(model, processor, device)

    def generate(self, prompt: str, num_images: int = 1, cfg_scale: float = 4.0,
                 num_timesteps: int = 30, resolution: int = 1024,
                 device: str = "cuda", **kwargs) -> List[Image.Image]:
        # Ming only generates one image at a time; loop for num_images > 1.
        images = []
        for _ in range(num_images):
            messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                audios=audio_inputs, return_tensors="pt",
            ).to(self.model.device)
            for k in inputs:
                if k in ("pixel_values", "pixel_values_videos", "audio_feats"):
                    inputs[k] = inputs[k].to(torch.bfloat16)
            image = self.model.generate(
                **inputs, image_gen=True,
                image_gen_cfg=cfg_scale, image_gen_steps=num_timesteps,
                image_gen_width=480, image_gen_height=544,
            )
            images.append(image)
        return images

    def get_transformer_layers(self) -> List[nn.Module]:
        return self.model.model.model.layers

    # ------------------------------------------------------------------ #
    #  Ming-specific pruning: module auto-accumulates when sparse_mode set  #
    # ------------------------------------------------------------------ #

    def setup_pruning_calibration(self, target_layer: str = "mlp") -> None:
        if target_layer != "mlp":
            return
        for layer in self.get_transformer_layers():
            for unit in self.get_mlp_units(layer, "und"):
                unit.sparse_mode = "prune"
        # Ming modules accumulate act_sum / act_cnt internally.

    def apply_pruning(
        self,
        keep_ratio: float,
        compressed_layers_und: Optional[List[int]],
        compressed_layers_gen: Optional[List[int]] = None,
        target_layer: str = "mlp",
        scores: Optional[dict] = None,
    ) -> None:
        if target_layer != "mlp":
            return
        from compress_utils import prune_mlp
        for layer in self.get_transformer_layers():
            for unit in self.get_mlp_units(layer, "und"):
                # Ming always prunes all layers (pass compressed_layers=None).
                prune_mlp(unit, keep_ratio, compressed_layers=None)
                unit.sparse_mode = "dense"
