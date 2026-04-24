# SPDX-License-Identifier: Apache-2.0
import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .base import BaseGenModel


class BagelGenModel(BaseGenModel):

    def __init__(self, model, vae_model, tokenizer, new_token_ids, device: str):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.device = device

    @classmethod
    def load(cls, model_path: str, device: str, tag: str = "ema",
             keep_ratio: float = 1.0, max_latent_size: int = 64, **kwargs) -> "BagelGenModel":
        from safetensors.torch import load_file
        from accelerate import init_empty_weights
        from data.data_utils import add_special_tokens
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.autoencoder import load_ae

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        setattr(llm_config, "keep_ratio", keep_ratio)

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= (1 if vit_config.num_hidden_layers == 27 else 0)

        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True, visual_und=True,
            llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
            vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
            latent_patch_size=2, max_latent_size=max_latent_size,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model = model.to(torch.bfloat16)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        state_dict = load_file(os.path.join(model_path, f"{tag}.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=False, assign=True)
        model = model.to(torch.bfloat16).to(device).eval()
        del state_dict

        vae_model = vae_model.to(device).eval()

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        return cls(model, vae_model, tokenizer, new_token_ids, device)

    def generate(self, prompt: str, num_images: int = 1, cfg_scale: float = 4.0,
                 num_timesteps: int = 30, resolution: int = 1024,
                 device: str = "cuda", strategy: str = "mask", **kwargs) -> List[Image.Image]:
        from modeling.bagel.qwen2_navit import NaiveCache
        from compress_utils import move_generation_input_to_device

        device = device or self.device
        past_kv = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        newlens = [0] * num_images
        new_rope = [0] * num_images

        gen_in, newlens, new_rope = self.model.prepare_prompts(
            curr_kvlens=newlens, curr_rope=new_rope,
            prompts=[prompt] * num_images,
            tokenizer=self.tokenizer, new_token_ids=self.new_token_ids,
        )
        gen_in = move_generation_input_to_device(gen_in, device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_kv = self.model.forward_cache_update_text(past_kv, **gen_in)

        gen_in = self.model.prepare_vae_latent(
            curr_kvlens=newlens, curr_rope=new_rope,
            image_sizes=[(resolution, resolution)] * num_images,
            new_token_ids=self.new_token_ids,
        )
        gen_in = move_generation_input_to_device(gen_in, device)

        cfg_past_kv = NaiveCache(self.model.config.llm_config.num_hidden_layers)
        cfg_newlens = [0] * num_images
        cfg_new_rope = [0] * num_images
        cfg_in = self.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_newlens, curr_rope=cfg_new_rope,
            image_sizes=[(resolution, resolution)] * num_images,
        )
        cfg_in = move_generation_input_to_device(cfg_in, device)

        gen_fn = self.model.generate_image_sparse if strategy == "mask" else self.model.generate_image
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_fn(
                past_key_values=past_kv,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=kwargs.get("cfg_interval", [0, 1.0]),
                cfg_renorm_min=kwargs.get("cfg_renorm_min", 0.0),
                timestep_shift=kwargs.get("timestep_shift", 3.0),
                cfg_text_past_key_values=cfg_past_kv,
                cfg_text_packed_position_ids=cfg_in["cfg_packed_position_ids"],
                cfg_text_key_values_lens=cfg_in["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=cfg_in["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=cfg_in["cfg_packed_key_value_indexes"],
                **gen_in,
            )

        images = []
        for latent in unpacked_latent:
            latent = latent.reshape(1, resolution // 16, resolution // 16, 2, 2, 16)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, 16, resolution // 8, resolution // 8)
            decoded = self.vae_model.decode(latent.to(device))
            arr = ((decoded * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            images.append(Image.fromarray(arr))
        return images

    def get_transformer_layers(self) -> List[nn.Module]:
        return self.model.language_model.model.layers

    # ------------------------------------------------------------------ #
    #  BAGEL-specific pruning overrides                                    #
    # ------------------------------------------------------------------ #

    def has_dual_path(self) -> bool:
        return True

    def get_mlp(self, layer: nn.Module, path: str = "und") -> nn.Module:
        return layer.mlp_moe_gen if path == "gen" else layer.mlp

    def setup_pruning_calibration(self, target_layer: str = "mlp") -> None:
        for layer in self.get_transformer_layers():
            if target_layer == "mlp":
                for path in ("und", "gen"):
                    mlp = self.get_mlp(layer, path)
                    mlp.sparse_mode = "prune"
                    mlp.register_buffer("act_sum", torch.zeros(mlp.intermediate_size))
                    mlp.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))
            elif target_layer == "attn":
                attn = self.get_attn(layer)
                attn.sparse_mode = "prune"
                act_sum = torch.zeros(attn.num_heads).to(attn.q_proj.weight.device)
                attn.register_buffer("act_sum", act_sum)
                attn.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))

    def apply_pruning(
        self,
        keep_ratio: float,
        compressed_layers_und: Optional[List[int]],
        compressed_layers_gen: Optional[List[int]] = None,
        target_layer: str = "mlp",
        scores: Optional[dict] = None,
    ) -> None:
        from compress_utils import prune_mlp
        for i, layer in enumerate(self.get_transformer_layers()):
            score = scores[i] if scores is not None else None
            if target_layer == "mlp":
                prune_mlp(self.get_mlp(layer, "und"), keep_ratio, compressed_layers_und, score)
                self.get_mlp(layer, "und").sparse_mode = "dense"
                prune_mlp(self.get_mlp(layer, "gen"), keep_ratio, compressed_layers_gen, score)
                self.get_mlp(layer, "gen").sparse_mode = "dense"
            elif target_layer == "attn":
                if compressed_layers_und and i in compressed_layers_und or \
                   compressed_layers_gen and i in compressed_layers_gen:
                    self._prune_attn(layer, keep_ratio, compressed_layers_und, compressed_layers_gen)
                    self.get_attn(layer).sparse_mode = "dense"

    @torch.no_grad()
    def _prune_attn(self, layer, keep_ratio, compressed_layers_und, compressed_layers_gen):
        attn = self.get_attn(layer)
        H_q = attn.num_heads
        H_kv = getattr(attn, "num_key_value_heads", H_q)
        groups = H_q // H_kv
        head_dim = attn.head_dim
        device = attn.q_proj.weight.device

        act_mean = attn.act_sum.to(device) / attn.act_cnt
        keep_q = torch.topk(act_mean, max(1, int(H_q * keep_ratio))).indices.sort().values
        prune_q = torch.tensor([h for h in range(H_q) if h not in keep_q], device=device)

        if prune_q.numel() > 0:
            mask = torch.ones(H_q * head_dim, device=device)
            for h in prune_q:
                mask[h * head_dim: (h + 1) * head_dim] = 0.0
            if compressed_layers_und and layer.self_attn.layer_idx in compressed_layers_und:
                attn.o_proj.weight.data.mul_(mask.unsqueeze(0))
            if compressed_layers_gen and layer.self_attn.layer_idx in compressed_layers_gen:
                attn.o_proj_moe_gen.weight.data.mul_(mask.unsqueeze(0))
