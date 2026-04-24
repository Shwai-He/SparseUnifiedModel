# SPDX-License-Identifier: Apache-2.0
"""Unified layer-dropping eval script for BAGEL, Ming, and Qwen-Image.

Replaces gen_images_mp_ld.py / gen_images_mp_ming_ld.py / gen_images_mp_qwen_ld.py.

Usage example (BAGEL):
  torchrun --nproc_per_node=8 gen_images_ld.py \
      --model_type bagel --model-path hf/BAGEL-7B-MoT/ \
      --output_dir out/ --metadata_file prompts.jsonl \
      --keep_ratio 0.75 --drop_type block --skip_mode und

Usage example (Ming / Qwen – same flags, skip_mode ignored):
  torchrun --nproc_per_node=1 gen_images_ld.py \
      --model_type ming --model-path your_model_path \
      --output_dir out/ --metadata_file prompts.jsonl \
      --keep_ratio 0.75 --drop_type block
"""

import os
import json
import argparse

import torch
import torch.distributed as dist

from models import load_model
from compress_utils import (
    setup_distributed, set_seed, get_transformer_layers, layer_drop_compress,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["bagel", "ming", "qwen"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="hf/BAGEL-7B-MoT/")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--calibration_samples", type=int, default=1)
    parser.add_argument("--drop_type", type=str, default="block", choices=["block", "attn", "mlp"])
    parser.add_argument("--skip_mode", type=str, default="und",
                        help="BAGEL only: 'und' or 'gen' modality to apply layer drop to")
    parser.add_argument("--strategy", type=str, default="mask", help="BAGEL only: mask | weight")
    parser.add_argument("--tag", type=str, default="ema", help="BAGEL only: safetensors tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_metadatas", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Qwen doesn't support multi-GPU in the original script; keep single-GPU for it.
    if args.model_type == "qwen":
        rank, world_size = 0, 1
        device = "cuda"
    else:
        setup_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"

    os.makedirs(args.output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {args.output_dir}")

    # --- Model loading ---
    model = load_model(
        args.model_type, args.model_path, device,
        tag=args.tag, max_latent_size=args.max_latent_size,
    )
    if rank == 0:
        n_params = sum(p.numel() for p in model.model.parameters() if hasattr(model, "model")) / 1e9
        print(f"Model loaded. Approx params: {n_params:.2f}B")

    # --- Metadata ---
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total = len(metadatas) if args.total_metadatas is None else args.total_metadatas

    prompts_per_gpu = (total + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total)

    # Common generation kwargs (model-specific ones are silently ignored in .generate())
    gen_kwargs = dict(
        cfg_scale=args.cfg_scale, num_timesteps=30,
        resolution=args.resolution, device=device, strategy=args.strategy,
    )

    # --- Layer-drop calibration ---
    def run_calibration():
        for idx in range(args.calibration_samples):
            prompt = metadatas[idx]["prompt"]
            for _ in range(args.num_images // args.batch_size):
                model.generate(prompt=prompt, num_images=args.batch_size, **gen_kwargs)

    layers = model.get_transformer_layers()
    # skip_mode is BAGEL-only; pass None for Ming/Qwen
    skip_mode = args.skip_mode if args.model_type == "bagel" else None
    layer_drop_compress(layers, run_calibration, args.keep_ratio, args.drop_type, skip_mode)

    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    # --- Generation loop ---
    for idx in range(start, end):
        metadata = metadatas[idx]
        outpath = os.path.join(args.output_dir, f"{idx:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata["prompt"]
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        if all(os.path.exists(os.path.join(sample_path, f"{i:05}.png")) for i in range(args.num_images)):
            print(f"GPU {rank} skipping (already done): {prompt}")
            continue

        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

        image_list = []
        for _ in range(args.num_images // args.batch_size):
            image_list.extend(model.generate(prompt=prompt, num_images=args.batch_size, **gen_kwargs))

        for i, img in enumerate(image_list):
            img.crop(img.getbbox()).save(os.path.join(sample_path, f"{i:05}.png"))

    print(f"GPU {rank} has completed all tasks")
    if args.model_type != "qwen":
        dist.barrier()
