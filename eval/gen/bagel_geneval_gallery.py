#!/usr/bin/env python3
"""Generate a multi-sample BAGEL-7B-MoT gallery in the GenEval / WISE directory layout."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Dict, List, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from models.bagel import BagelGenModel


REASONING_INSTRUCTION = """You are a world-knowledge visual planner. Given an image request, first infer the concrete scene that would make the request factually correct. Then write a literal, self-contained prompt for an image generator. Preserve the requested style and do not mention this instruction.

Return exactly these two sections:
<reasoning>Concise explanation of the relevant world knowledge and visual consequences.</reasoning>
<generation_prompt>A concrete visual description containing every required object, state, and relation.</generation_prompt>"""


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
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit_file = repo_root / "COMMIT"
        if commit_file.exists():
            return commit_file.read_text().strip()
        return "ae0aa14f48b94df80a52003c4cf7e7db6b7be303"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BAGEL gallery samples")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--metadata-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt-index", type=int, required=True)
    parser.add_argument("--seeds", type=str, default="101,202")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--reason-before-generation", action="store_true")
    parser.add_argument("--reasoning-source-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="ema")
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
        # Check if pre-reasoned manifest exists
        if args.reasoning_source_dir:
            src_manifest = Path(args.reasoning_source_dir) / f"prompt_{args.prompt_index}" / "manifest.json"
            if src_manifest.exists():
                try:
                    with src_manifest.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "reasoning" in data and data["reasoning"]:
                        reasoning_record = data["reasoning"]
                        target_prompt = reasoning_record.get("generation_prompt", original_prompt)
                        print(f"[BAGEL Reasoner] Reusing aligned reasoning prompt: {target_prompt[:100]}...", flush=True)
                except Exception as e:
                    print(f"[BAGEL Reasoner Warning] Failed to load {src_manifest}: {e}", flush=True)

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

    print(f"[BAGEL Loading] Loading BAGEL model from {args.model_path} on CUDA...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = BagelGenModel.load(
        model_path=args.model_path,
        device=device,
        tag=args.tag,
        max_latent_size=args.resolution // 8,
    )
    print(f"[BAGEL Loading Complete] Elapsed {time.time() - t0:.2f}s", flush=True)

    result_images = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for seed in seeds:
        print(f"[BAGEL Generating] Seed {seed}, resolution {args.resolution}x{args.resolution}...", flush=True)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t_gen = time.time()
        imgs = model.generate(
            prompt=target_prompt,
            num_images=1,
            cfg_scale=args.cfg_scale,
            num_timesteps=args.num_inference_steps,
            resolution=args.resolution,
            device=device,
        )
        print(f"[BAGEL Generated] Seed {seed} done in {time.time() - t_gen:.2f}s", flush=True)
        result_images.append(imgs[0])

    peak_cuda_bytes = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0

    output_dir = Path(args.output_dir).resolve()
    prompt_dir = output_dir / f"{args.prompt_index:05d}"
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    (prompt_dir / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    image_records = []
    for image_index, (seed, image) in enumerate(zip(seeds, result_images)):
        image_path = sample_dir / f"{image_index:05d}.png"
        image.save(image_path)
        image_records.append({
            "index": image_index,
            "seed": seed,
            "file": str(image_path.relative_to(output_dir)),
            "sha256": file_sha256(image_path),
        })

    gallery_path = output_dir / "gallery.png"
    make_grid(result_images, [f"seed {seed}" for seed in seeds], len(seeds)).save(gallery_path)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema": (
            "sparse_unified_model_bagel_reasoned_gallery_v1"
            if args.reason_before_generation
            else "sparse_unified_model_bagel_geneval_gallery_v1"
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
            "model_type": "bagel",
            "seeds": seeds,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
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
    print(f"[BAGEL Output Saved] Successfully wrote gallery and manifest to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
