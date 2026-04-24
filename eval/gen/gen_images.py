# SPDX-License-Identifier: Apache-2.0
"""Unified weight-pruning eval script for BAGEL, Ming, and Qwen-Image.

Replaces gen_images_mp.py / gen_images_mp_ming.py / gen_images_mp_qwen.py.

Usage example (BAGEL, MLP prune):
  torchrun --nproc_per_node=8 gen_images.py \
      --model_type bagel --model-path hf/BAGEL-7B-MoT/ \
      --output_dir out/ --metadata_file prompts.jsonl \
      --keep_ratio 0.75 --sparse_mode prune --target_layer mlp \
      --compressed_layers_und 0-28 --compressed_layers_gen 0-28

Usage example (Ming / Qwen):
  torchrun --nproc_per_node=1 gen_images.py \
      --model_type ming --model-path your_model_path \
      --output_dir out/ --metadata_file prompts.jsonl \
      --keep_ratio 0.75 --sparse_mode prune
"""

import os
import json
import argparse

import torch
import torch.distributed as dist

from models import load_model
from compress_utils import (
    setup_distributed, set_seed, parse_layer_range,
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
    parser.add_argument("--sparse_mode", type=str, default="prune", choices=["prune", "random"])
    parser.add_argument("--target_layer", type=str, default="mlp", choices=["mlp", "attn"],
                        help="BAGEL only: 'attn' enables head pruning instead of MLP pruning")
    parser.add_argument("--compressed_layers_und", type=str, default="0-0",
                        help="Layer range for und path, e.g. '0-28' or '0-28-2'")
    parser.add_argument("--compressed_layers_gen", type=str, default="0-0",
                        help="Layer range for gen path (BAGEL only)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to pre-computed importance scores (.pt file)")
    parser.add_argument("--strategy", type=str, default="weight")
    parser.add_argument("--tag", type=str, default="ema", help="BAGEL only: safetensors tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_metadatas", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

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

    # --- Metadata ---
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]
    total = len(metadatas) if args.total_metadatas is None else args.total_metadatas

    prompts_per_gpu = (total + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total)

    compressed_layers_und = parse_layer_range(args.compressed_layers_und)
    # For non-BAGEL models the gen path doesn't exist; pass the same list as und.
    compressed_layers_gen = (
        parse_layer_range(args.compressed_layers_gen)
        if args.model_type == "bagel"
        else compressed_layers_und
    )

    gen_kwargs = dict(
        cfg_scale=args.cfg_scale, num_timesteps=30,
        resolution=args.resolution, device=device, strategy=args.strategy,
    )

    # --- Weight-pruning calibration + application ---
    if args.keep_ratio < 1.0 and args.strategy == "weight":
        scores = torch.load(args.cache_dir) if args.cache_dir else None

        if args.sparse_mode == "random":
            # Random sparsity: set mode on each MLP unit, no calibration needed.
            for layer in model.get_transformer_layers():
                for path in (["und", "gen"] if model.has_dual_path() else ["und"]):
                    in_list = compressed_layers_und if path == "und" else compressed_layers_gen
                    for unit in model.get_mlp_units(layer, path):
                        if not in_list or (hasattr(unit, "layer_idx") and unit.layer_idx in in_list):
                            unit.sparse_mode = "random"
                            unit.sparsity_ratio = args.keep_ratio

        elif args.sparse_mode == "prune":
            model.setup_pruning_calibration(args.target_layer)

            if scores is None:
                for idx in range(args.calibration_samples):
                    prompt = metadatas[idx]["prompt"]
                    sample_path = os.path.join(args.output_dir, f"{idx:0>5}", "samples")
                    os.makedirs(sample_path, exist_ok=True)
                    for _ in range(args.num_images // args.batch_size):
                        model.generate(prompt=prompt, num_images=args.batch_size, **gen_kwargs)

            model.apply_pruning(
                keep_ratio=args.keep_ratio,
                compressed_layers_und=compressed_layers_und,
                compressed_layers_gen=compressed_layers_gen,
                target_layer=args.target_layer,
                scores=scores,
            )

    total_params = sum(p.numel() for p in model.get_transformer_layers()[0].parameters()) if model.get_transformer_layers() else 0
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
