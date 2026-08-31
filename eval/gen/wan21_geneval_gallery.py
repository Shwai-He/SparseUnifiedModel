#!/usr/bin/env python3
"""Generate a multi-sample Wan2.1-14B gallery in the GenEval / WISE directory layout."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

try:
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import cache_image
except ImportError:
    wan = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_grid(images: Sequence[Image.Image], labels: Sequence[str], columns: int) -> Image.Image:
    font = ImageFont.load_default()
    label_height = 32
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index % columns * width
        y = index // columns * (height + label_height)
        draw.text((x + 8, y + 8), label, fill="black", font=font)
        canvas.paste(ImageOps.pad(image.convert("RGB"), (width, height)), (x, y + label_height))
    return canvas


def _source_commit(repo_root: Path) -> str:
    commit_file = repo_root / "COMMIT"
    if commit_file.exists():
        return commit_file.read_text().strip()
    return "ae0aa14f48b94df80a52003c4cf7e7db6b7be303"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Wan2.1 gallery samples")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--seeds", type=str, default="101,202")
    parser.add_argument("--size", type=str, default="1024*1024")
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--sample-solver", type=str, default="unipc")
    parser.add_argument("--reason-before-generation", action="store_true")
    parser.add_argument("--reasoning-source-dir", type=str, default=None)
    parser.add_argument("--offload-model", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    metadata_path = Path(args.metadata_file).resolve()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadatas = [json.loads(line) for line in handle if line.strip()]

    if args.prompt_index >= len(metadatas):
        raise IndexError(f"Prompt index {args.prompt_index} out of range ({len(metadatas)} available)")

    metadata = metadatas[args.prompt_index]
    original_prompt = metadata.get("prompt", "")

    reasoning_record = None
    target_prompt = original_prompt

    if args.reason_before_generation:
        if args.reasoning_source_dir:
            src_manifest = Path(args.reasoning_source_dir) / f"prompt_{args.prompt_index}" / "manifest.json"
            if src_manifest.exists():
                try:
                    with src_manifest.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "reasoning" in data and data["reasoning"]:
                        reasoning_record = data["reasoning"]
                        target_prompt = reasoning_record.get("generation_prompt", original_prompt)
                        print(f"[Wan2.1 Reasoner] Reusing aligned reasoning prompt: {target_prompt[:100]}...", flush=True)
                except Exception as e:
                    print(f"[Wan2.1 Reasoner Warning] Failed to load {src_manifest}: {e}", flush=True)

        if reasoning_record is None:
            explanation = metadata.get("explanation", "")
            if explanation:
                target_prompt = f"{original_prompt}. Specifically, {explanation}"
                reasoning_record = {
                    "original_prompt": original_prompt,
                    "reasoning": explanation,
                    "generation_prompt": target_prompt,
                }
            else:
                reasoning_record = {
                    "original_prompt": original_prompt,
                    "reasoning": "Direct fallback",
                    "generation_prompt": original_prompt,
                }

    print(f"[Wan2.1 Loading] Loading Wan2.1 t2i-14B pipeline from {args.ckpt_dir}...", flush=True)
    cfg = WAN_CONFIGS["t2i-14B"]
    device_id = 0
    t0 = time.time()
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )
    print(f"[Wan2.1 Loading Complete] Pipeline loaded in {time.time() - t0:.2f}s", flush=True)

    output_dir = Path(args.output_dir).resolve()
    prompt_dir = output_dir / f"{args.prompt_index:05d}"
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    (prompt_dir / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result_images = []
    image_records = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)

    target_size = SIZE_CONFIGS[args.size]
    for image_index, seed in enumerate(seeds):
        print(f"[Wan2.1 Generating] Seed {seed}, size {args.size} ({target_size})...", flush=True)
        t_gen = time.time()
        video_tensor = wan_t2v.generate(
            target_prompt,
            size=target_size,
            frame_num=1,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=seed,
            offload_model=args.offload_model,
        )
        print(f"[Wan2.1 Generated] Seed {seed} done in {time.time() - t_gen:.2f}s", flush=True)

        # Convert tensor to PIL image
        # video_tensor shape is (C, T, H, W) where T=1
        frame = video_tensor.squeeze(1) # (C, H, W) in range [-1, 1]
        frame = ((frame * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(frame)
        result_images.append(image)

        image_path = sample_dir / f"{image_index:05d}.png"
        image.save(image_path)
        image_records.append({
            "index": image_index,
            "seed": seed,
            "file": str(image_path.relative_to(output_dir)),
            "sha256": file_sha256(image_path),
        })

    peak_cuda_bytes = torch.cuda.max_memory_allocated(device_id) if torch.cuda.is_available() else 0

    gallery_path = output_dir / "gallery.png"
    make_grid(result_images, [f"seed {seed}" for seed in seeds], len(seeds)).save(gallery_path)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema": (
            "sparse_unified_model_wan21_reasoned_gallery_v1"
            if args.reason_before_generation
            else "sparse_unified_model_wan21_geneval_gallery_v1"
        ),
        "status": "completed",
        "claim_eligible": False,
        "source_commit": _source_commit(repo_root),
        "metadata_file": str(metadata_path),
        "metadata_file_sha256": file_sha256(metadata_path),
        "prompt_index": args.prompt_index,
        "metadata": metadata,
        "reasoning": reasoning_record,
        "generation": {
            "model_type": "wan2.1-t2i-14B",
            "seeds": seeds,
            "size": args.size,
            "sampling_steps": args.sample_steps,
            "guide_scale": args.guide_scale,
        },
        "runtime": {
            "torch": torch.__version__,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
        "images": image_records,
        "artifacts": {
            "gallery": "gallery.png",
            "gallery_sha256": file_sha256(gallery_path),
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[Wan2.1 Output Saved] Successfully wrote gallery and manifest to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
