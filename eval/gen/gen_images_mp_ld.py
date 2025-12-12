# SPDX-License-Identifier: Apache-2.0

import os
import json
import pickle
import argparse
from safetensors.torch import load_file

import torch
import torch.nn.functional as F
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


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None, strategy="mask"):  # 添加device参数
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
            if strategy == "mask": 
                unpacked_latent = gen_model.generate_image_sparse(
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
            else: 
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



def fewshot_compress(keep_ratio, calibration_samples, sparse_mode="prune", drop_type="block", record=False, skip_mode="und"): 
   

    record=True


    for i, layer in enumerate(model.language_model.model.layers):
        layer.record = record
        layer.skip_mode = skip_mode

    for idx in range(calibration_samples):

        metadata = metadatas[idx]
        outpath = os.path.join(output_dir, f"{idx:0>5}")
        prompt = metadata['prompt']

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
                strategy=args.strategy,
            )
    
    COS_SIM = {}
    TYPES = ["block", "attn", "mlp"]
    for Type in TYPES:
        COS_SIM[Type] = [] 

    for i, layer in enumerate(model.language_model.model.layers):
        # block
        cos_sim = F.cosine_similarity(layer.input, layer.output, dim=-1)  # (total_token_num)
        cos_sim = cos_sim.mean()
        COS_SIM["block"].append(cos_sim.cpu().data.item())
        # attn
        cos_sim = F.cosine_similarity(layer.input, layer.residual, dim=-1)  # (total_token_num)
        cos_sim = cos_sim.mean()
        COS_SIM["attn"].append(cos_sim.cpu().data.item())
        # mlp
        cos_sim = F.cosine_similarity(layer.residual, layer.output, dim=-1)  # (total_token_num)
        cos_sim = cos_sim.mean()
        COS_SIM["mlp"].append(cos_sim.cpu().data.item())

    COS_SIM_TYPE = COS_SIM[drop_type]
    print(f"COS_SIM_TYPE: {COS_SIM_TYPE}")
    topk_weight, topk_index = torch.topk(torch.tensor(COS_SIM_TYPE), int((1 - keep_ratio) * len(COS_SIM_TYPE)), dim=-1)
    
    # if "und" in skip_mode:
    #     topk_index = range(int((1 - keep_ratio) * len(model.language_model.model.layers)), len(model.language_model.model.layers))
    # else:
    #     topk_index = range(0, int((1 - keep_ratio) * len(model.language_model.model.layers)))


    for i, layer in enumerate(model.language_model.model.layers):
        layer.skip_mode = skip_mode
        if i in topk_index:
            layer.skip_type = drop_type
        else: 
            layer.skip_type = None
        layer.record = False
        del layer.input
        del layer.residual
        del layer.output    


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
    parser.add_argument('--drop_type', type=str, default="block") #######
    parser.add_argument('--tag', type=str, default='ema')
    parser.add_argument('--record_only', type=str, default=False)
    parser.add_argument('--strategy', type=str, default="weight")
    parser.add_argument('--skip_mode', type=str, default="und")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_metadatas', type=int, default=None)

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

    model_state_dict_path = os.path.join(args.model_path, f"{args.tag}.safetensors")
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

    fewshot_compress(args.keep_ratio, args.calibration_samples, args.sparse_mode, args.drop_type, skip_mode=args.skip_mode)

    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

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
                strategy=args.strategy,
            )
            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()