# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
import pickle
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration

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


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None, strategy="mask", seed=42):  # 添加device参数

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_cfg=cfg_scale, 
        image_gen_steps=num_timesteps, 
        image_gen_width=480, 
        image_gen_height=544
    )

    return [image]


def fewshot_compress(keep_ratio, calibration_samples, sparse_mode="prune", drop_type="block", record=False): 
    
    record=True
    for i, layer in enumerate(model.model.model.layers):
        layer.record = record

    # for idx in range(calibration_samples):

    #     metadata = metadatas[idx]
    #     outpath = os.path.join(output_dir, f"{idx:0>5}")
    #     prompt = metadata['prompt']

    #     for i in range(args.num_images // args.batch_size):
    #         generate_image(
    #             prompt=prompt,
    #             cfg_scale=cfg_scale, 
    #             cfg_interval=cfg_interval, 
    #             cfg_renorm_min=cfg_renorm_min,
    #             timestep_shift=timestep_shift, 
    #             num_timesteps=num_timesteps,
    #             num_images=args.batch_size,
    #             resolution=args.resolution,
    #             device=device,
    #             strategy=args.strategy,
    #         )
    
    # COS_SIM = {}
    # TYPES = ["block", "attn", "mlp"]
    # for Type in TYPES:
    #     COS_SIM[Type] = [] 

    # for i, layer in enumerate(model.model.model.layers):
    #     # block
    #     cos_sim = F.cosine_similarity(layer.input, layer.output, dim=-1)  # (total_token_num)
    #     cos_sim = cos_sim.mean()
    #     COS_SIM["block"].append(cos_sim.cpu().data.item())
    #     # attn
    #     cos_sim = F.cosine_similarity(layer.input, layer.residual, dim=-1)  # (total_token_num)
    #     cos_sim = cos_sim.mean()
    #     COS_SIM["attn"].append(cos_sim.cpu().data.item())
    #     # mlp
    #     cos_sim = F.cosine_similarity(layer.residual, layer.output, dim=-1)  # (total_token_num)
    #     cos_sim = cos_sim.mean()
    #     COS_SIM["mlp"].append(cos_sim.cpu().data.item())

    # COS_SIM_TYPE = COS_SIM[drop_type]
    # topk_weight, topk_index = torch.topk(torch.tensor(model.model.model.layers), int((1 - keep_ratio) * len(model.model.model.layers)), dim=-1)
    topk_index = range(int(keep_ratio * len(model.model.model.layers)), len(model.model.model.layers))

    for i, layer in enumerate(model.model.model.layers):
        if i in topk_index:
            layer.skip_mode = drop_type
        else: 
            layer.skip_mode = None
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
    parser.add_argument('--drop_type', type=str, default="block")
    parser.add_argument('--tag', type=str, default='ema')
    parser.add_argument('--record_only', type=str, default=False)
    parser.add_argument('--strategy', type=str, default="weight")
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

    rank = 0
    world_size = 1

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    model_path = "/mnt/bn/seed-aws-va/shwai.he/models/inclusionAI/Ming-Lite-Omni-1.5"

    torch_dtype = torch.bfloat16
    device = "cuda"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        load_image_gen=True,
        low_cpu_mem_usage=True       # Minimize CPU memory during loading
    )
    # .to("cuda")  

    model = model.to(device)
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)

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

    fewshot_compress(args.keep_ratio, args.calibration_samples, args.sparse_mode, args.drop_type)

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
                seed=seed,
            )
            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()