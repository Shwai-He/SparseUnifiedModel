# SPDX-License-Identifier: Apache-2.0

import os
import json
import pickle
import argparse
from safetensors.torch import load_file

import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from tqdm import tqdm
from modeling.bagel.qwen2_navit import NaiveCache

from accelerate import init_empty_weights, dispatch_model


def print_memory(prefix=""):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    
    print(f"[{prefix}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Reserved: {max_reserved:.2f} MB")
    print()

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None):
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)] * num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


@torch.no_grad()
def prune(
            layer, 
            keep_ratio: float = 0.5, 
            compressed_layers: list = [], 
            score: torch.Tensor = None,
            ):

    if score is None:
        act_mean = layer.act_sum / layer.act_cnt
        col_norm = layer.down_proj.weight.norm(dim=0).cpu()
        score    = act_mean * col_norm

    k = int(layer.intermediate_size * keep_ratio)
    keep = torch.topk(score, k).indices.sort().values      # ascending order
    
    if layer.layer_idx in compressed_layers:
        layer.gate_proj.weight.data = layer.gate_proj.weight.data[keep]
        layer.up_proj.weight.data   = layer.up_proj.weight.data[keep]
        layer.down_proj.weight.data = layer.down_proj.weight.data[:, keep]
        layer.intermediate_size      = k
        layer.gate_proj.out_features = k
        layer.up_proj.out_features   = k
        layer.down_proj.in_features  = k

        if hasattr(layer, "act_sum"):
            del layer.act_sum
        if hasattr(layer, "act_cnt"):
            del layer.act_cnt
    
    return keep.tolist()


@torch.no_grad()
def prune_attn(
    layer,
    keep_ratio: float = 0.5,
    use_group: bool = False,
    score: torch.Tensor = None,
    compressed_layers_und: list = [], 
    compressed_layers_gen: list = [],    
    ):

    H_q  = layer.num_heads
    H_kv = getattr(layer, "num_key_value_heads", H_q)
    groups = H_q // H_kv
    head_dim = layer.head_dim
    device = layer.q_proj.weight.device

    if score is None: 
        act_mean = layer.act_sum.to(device) / layer.act_cnt   # (H_q,)
    else:
        act_mean = score
    if use_group:
        score_group = act_mean.view(H_kv, groups).mean(dim=1)           # (H_kv,)
        keep_kv = torch.topk(score_group, max(1, int(H_kv * keep_ratio))).indices
        keep_q  = torch.cat([
            torch.arange(g*groups, (g+1)*groups, device=device) for g in keep_kv
        ])
    else:
        keep_q = torch.topk(act_mean, max(1, int(H_q * keep_ratio))).indices
    keep_q  = keep_q.sort().values
    prune_q = torch.tensor([i for i in range(H_q) if i not in keep_q], device=device)

    if prune_q.numel() > 0:
        mask = torch.ones(H_q * head_dim, device=device)
        for h in prune_q:
            mask[h*head_dim : (h+1)*head_dim] = 0.0

        if layer.layer_idx in compressed_layers_und:
            layer.o_proj.weight.data.mul_(mask.unsqueeze(0))  # broadcast 到行
        elif layer.layer_idx in compressed_layers_gen:
            layer.o_proj_moe_gen.weight.data.mul_(mask.unsqueeze(0))  # broadcast 到行

    return prune_q.cpu().tolist()


def fewshot_compress(   
                    keep_ratio, 
                    compressed_layers_und, 
                    compressed_layers_gen, 
                    calibration_samples, 
                    sparse_mode="prune", 
                    record=False, 
                    target_layer="mlp", 
                    cache_dir=None, 
                    ): 
    scores = None
    if sparse_mode == "random": 
        for i, layer in enumerate(model.language_model.model.layers):
            if layer.layer_idx in compressed_layers_und:
                layer.mlp.sparse_mode = "random"
                layer.mlp.sparsity_ratio = keep_ratio

            if layer.layer_idx in compressed_layers_gen:
                layer.mlp_moe_gen.sparse_mode = "random"
                layer.mlp_moe_gen.sparsity_ratio = keep_ratio
    
    elif sparse_mode == "prune":
        if cache_dir is None: 
            if target_layer == "mlp":
                for i, layer in enumerate(model.language_model.model.layers):
                    layer.mlp.sparse_mode = "prune"    
                    layer.mlp.register_buffer("act_sum", torch.zeros(layer.mlp.intermediate_size))     # ∑|h|
                    layer.mlp.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数
                    layer.mlp_moe_gen.sparse_mode = "prune"    
                    layer.mlp_moe_gen.register_buffer("act_sum", torch.zeros(layer.mlp_moe_gen.intermediate_size))     # ∑|h|
                    layer.mlp_moe_gen.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数

            elif target_layer == "attn":
                for l, layer in enumerate(model.language_model.model.layers):
                    if l in compressed_layers_und or l in compressed_layers_gen: 
                        layer.self_attn.sparse_mode = "prune"    
                        act_sum_tensor = torch.zeros(layer.self_attn.num_heads).to(layer.self_attn.q_proj.weight.device)
                        layer.self_attn.register_buffer("act_sum", act_sum_tensor)  # ∑|h|
                        layer.self_attn.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数
            else: 
                raise NotImplementedError

            for idx in range(calibration_samples):

                metadata = metadatas[idx]
                outpath = os.path.join(output_dir, f"{idx:0>5}")
                prompt = metadata['prompt']

                sample_path = os.path.join(outpath, "samples")
                os.makedirs(sample_path, exist_ok=True)

                for i in range(args.num_images // args.batch_size):
                    generate_image(
                        prompt=prompt,
                        cfg_scale=cfg_scale, 
                        cfg_interval=cfg_interval, 
                        cfg_renorm_min=cfg_renorm_min,
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        num_images=args.batch_size,
                        resolution=args.resolution,
                        device=device,
                    )
        else: 
            scores = torch.load(f"{cache_dir}")

        for l, layer in enumerate(model.language_model.model.layers):
            if target_layer == "mlp":
                prune(layer.mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers_und, score=scores[l] if scores is not None else None, )
                prune(layer.mlp_moe_gen, keep_ratio=keep_ratio, compressed_layers=compressed_layers_gen, score=scores[l] if scores is not None else None, )

                layer.mlp.sparse_mode = "dense"    
                layer.mlp_moe_gen.sparse_mode = "dense"    

            elif target_layer == "attn": 
                if l in compressed_layers_und or l in compressed_layers_gen:
                    prune_attn( 
                                layer.self_attn, 
                                keep_ratio=keep_ratio, 
                                score=scores[l] if scores is not None else None, 
                                compressed_layers_und=compressed_layers_und,
                                compressed_layers_gen=compressed_layers_gen,
                                )
                    layer.self_attn.sparse_mode = "dense"
                
            else: 
                raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--metadata_file", type=str, required=True, help="JSONL file containing lines of metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--keep_ratio', type=float, default=1.0)
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--compressed_layers_und', type=str, default="0-0")
    parser.add_argument('--compressed_layers_gen', type=str, default="0-0")
    parser.add_argument('--sparse_mode', type=str, default='prune')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_metadatas', type=int, default=None)
    parser.add_argument('--target_layer', type=str, default="mlp")
    parser.add_argument('--cache_dir', type=str, default=None)

    args = parser.parse_args()
    
    seed = args.seed
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    setattr(llm_config, "keep_ratio", args.keep_ratio)

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1 if vit_config.num_hidden_layers == 27 else vit_config.num_hidden_layers

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model = model.to(torch.bfloat16)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    if rank == 0:
        print(model)

    model_state_dict_path = os.path.join(args.model_path, f"ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")

    msg = model.load_state_dict(model_state_dict, strict=False, assign=True)

    model = model.to(torch.bfloat16)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    if rank == 0:
        print(msg)
        print(model.dtype)

    del model_state_dict

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    num_timesteps = 30
    cfg_renorm_min = 0.0

    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    if args.total_metadatas is None: 
        total_metadatas = len(metadatas)
    else: 
        total_metadatas = args.total_metadatas

    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    if len(args.compressed_layers_und.split('-')) == 2:
        compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1])))
    elif len(args.compressed_layers_und.split('-')) == 3:
        compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1]), int(args.compressed_layers_und.split('-')[2])))
    else: 
        NotImplementedError

    if len(args.compressed_layers_gen.split('-')) == 2:
        compressed_layers_gen = list(range(int(args.compressed_layers_gen.split('-')[0]), int(args.compressed_layers_gen.split('-')[1])))
    elif len(args.compressed_layers_gen.split('-')) == 3:
        compressed_layers_gen = list(range(int(args.compressed_layers_gen.split('-')[0]), int(args.compressed_layers_gen.split('-')[1]), int(args.compressed_layers_gen.split('-')[2])))
    else: 
        NotImplementedError

    setattr(llm_config, "compressed_layers_und", compressed_layers_und)
    setattr(llm_config, "compressed_layers_gen", compressed_layers_gen)

    if args.keep_ratio < 1.0:
        calibration_samples = args.calibration_samples
        sparse_mode = args.sparse_mode
        fewshot_compress(args.keep_ratio, 
                        compressed_layers_und, 
                        compressed_layers_gen, 
                        calibration_samples, 
                        sparse_mode, 
                        target_layer=args.target_layer,
                        cache_dir=args.cache_dir, 
                        )
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    for idx in range(start, end):
        metadata = metadatas[idx]
        outpath = os.path.join(output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        flag = True
        for idx in range(args.num_images):
            if not os.path.exists(os.path.join(sample_path, f"{idx:05}.png")):
                flag = False
                break
        if flag:
            print(f"GPU {rank} skipping generation for prompt: {prompt}")
            continue

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        image_list = []

        for i in range(args.num_images // args.batch_size):
            tmp_image_list = generate_image(
                prompt=prompt,
                cfg_scale=cfg_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                num_images=args.batch_size,
                resolution=args.resolution,
                device=device,
            )
            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()