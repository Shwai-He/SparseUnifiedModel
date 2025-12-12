# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
import pickle
import argparse
from tqdm import tqdm

import torch
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


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None, seed=42):

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

@torch.no_grad()
def prune(
            layer, 
            keep_ratio: float = 0.5, 
            compressed_layers: list = None, 
            ):

    if not hasattr(layer, "score"):
        act_mean = layer.act_sum / layer.act_cnt
        col_norm = layer.down_proj.weight.norm(dim=0).cpu()
        score    = act_mean * col_norm
        layer.score = score
    else: 
        score = layer.score

    k = int(layer.intermediate_size * keep_ratio)
    keep = torch.topk(score, k).indices.sort().values
    
    layer.gate_proj.weight.data = layer.gate_proj.weight.data[keep]
    layer.up_proj.weight.data   = layer.up_proj.weight.data[keep]
    layer.down_proj.weight.data = layer.down_proj.weight.data[:, keep]
    layer.intermediate_size      = k
    layer.gate_proj.out_features = k
    layer.up_proj.out_features   = k
    layer.down_proj.in_features  = k

    del layer.act_sum
    del layer.act_cnt

    return keep.tolist()


def fewshot_compress(keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode="prune", scores=None): 
   
    if sparse_mode == "random": 
        mlp = layer.mlp 
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                expert.sparse_mode = "random"
                expert.sparsity_ratio = keep_ratio
        else: 
            mlp.sparse_mode = "random"    
            mlp.sparsity_ratio = keep_ratio    
        return 

    elif sparse_mode == "prune":
        for i, layer in enumerate(model.model.model.layers):
            mlp = layer.mlp 
            if hasattr(mlp, "experts"):
                for expert in mlp.experts:
                    expert.sparse_mode = "prune"
            else: 
                mlp.sparse_mode = "prune"    

        if scores is None: 
            for idx in range(calibration_samples):
                metadata = metadatas[idx]
                outpath = os.path.join(output_dir, f"{idx:0>5}")
                prompt = metadata['prompt']

                sample_path = os.path.join(outpath, "samples")
                os.makedirs(sample_path, exist_ok=True)

                for i in range(args.num_images // args.batch_size):
                    if len(compressed_layers_gen) == 0: 
                        num_timesteps = 2

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
                        seed=seed,
                    )

        target_layer = "mlp"
        compressed_layers = []
        for i, layer in enumerate(model.model.model.layers):
            mlp = layer.mlp 
            if hasattr(mlp, "experts"):
                for expert in mlp.experts:
                    if target_layer == "mlp":
                        keep_und = prune(expert, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
            else: 
                if target_layer == "mlp":
                    keep_und = prune(mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)

        for i, layer in enumerate(model.model.model.layers):
            mlp = layer.mlp 
            if hasattr(mlp, "experts"):
                for expert in mlp.experts:
                    expert.sparse_mode = "dense"
            else: 
                mlp.sparse_mode = "dense" 


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
    parser.add_argument('--tag', type=str, default='ema')
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

    model_path = "your_model_path"

    torch_dtype = torch.bfloat16
    device = "cuda"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        load_image_gen=True,
        low_cpu_mem_usage=True
    )

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

    if args.keep_ratio < 1.0:
        calibration_samples = args.calibration_samples
        sparse_mode = args.sparse_mode
        fewshot_compress(args.keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode)

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