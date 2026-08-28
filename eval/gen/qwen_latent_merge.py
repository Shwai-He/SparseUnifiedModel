#!/usr/bin/env python3
"""Compare two Qwen-Image seeds with merged intermediate DiT latents."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from models import load_model


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    value = tensor.detach().float()
    return {
        "l2_norm": torch.linalg.vector_norm(value).item(),
        "rms": torch.sqrt(torch.mean(value.square())).item(),
    }


def slerp(left: torch.Tensor, right: torch.Tensor, alpha: float) -> torch.Tensor:
    left_flat = left.float().reshape(left.shape[0], -1)
    right_flat = right.float().reshape(right.shape[0], -1)
    left_norm = torch.linalg.vector_norm(left_flat, dim=1, keepdim=True).clamp_min(1e-8)
    right_norm = torch.linalg.vector_norm(right_flat, dim=1, keepdim=True).clamp_min(1e-8)
    cosine = ((left_flat / left_norm) * (right_flat / right_norm)).sum(dim=1, keepdim=True)
    cosine = cosine.clamp(-0.9995, 0.9995)
    angle = torch.acos(cosine)
    denominator = torch.sin(angle).clamp_min(1e-8)
    merged = torch.sin((1.0 - alpha) * angle) / denominator * left_flat
    merged += torch.sin(alpha * angle) / denominator * right_flat
    return merged.reshape_as(left).to(dtype=left.dtype)


def merge_latents(left: torch.Tensor, right: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if method == "lerp":
        return torch.lerp(left, right, alpha)
    if method == "slerp":
        return slerp(left, right, alpha)
    raise ValueError(f"unsupported merge method: {method}")


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


def _encoded_prompt(pipe, prompt: str, max_sequence_length: int):
    encoded = pipe.encode_prompt(
        prompt=[prompt],
        device=pipe._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return encoded[0], encoded[1]


def _source_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _load_model(args: argparse.Namespace):
    pipeline_kwargs = {}
    text_encoder_kwargs = {}
    precision = getattr(args, "precision", "nf4")
    load_on_cpu = False
    sequential_cpu_offload = False
    if precision == "nf4":
        from diffusers import PipelineQuantizationConfig
        from transformers import BitsAndBytesConfig

        pipeline_kwargs.update(
            quantization_config=PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer"],
            ),
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
        text_encoder_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif precision == "bf16":
        pipeline_kwargs["low_cpu_mem_usage"] = True
        text_encoder_kwargs.update(torch_dtype=torch.bfloat16, device_map={"": "cpu"})
        load_on_cpu = True
    else:
        raise ValueError(f"unsupported precision: {precision}")
    model = load_model(
        "qwen",
        args.model_path,
        "cuda",
        pipeline_kwargs=pipeline_kwargs,
        text_encoder_kwargs=text_encoder_kwargs,
        enable_model_cpu_offload=precision == "nf4",
        enable_sequential_cpu_offload=sequential_cpu_offload,
        load_on_cpu=load_on_cpu,
    )
    if args.lora_path:
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = dict(model.pipe.scheduler.config)
        scheduler_config.update(
            base_shift=math.log(3),
            max_shift=math.log(3),
            shift=1.0,
            max_image_seq_len=8192,
            stochastic_sampling=False,
            time_shift_type="exponential",
            use_dynamic_shifting=True,
        )
        model.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        model.pipe.load_lora_weights(args.lora_path, weight_name=args.lora_weight)
    return model


def run(args: argparse.Namespace) -> Path:
    model = _load_model(args)
    pipe = model.pipe
    if args.low_memory:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    device = pipe._execution_device
    prompt_embeds, prompt_mask = _encoded_prompt(pipe, args.prompt, args.max_sequence_length)
    if prompt_mask is None:
        prompt_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=device)
    negative_prompt_embeds = None
    negative_prompt_mask = None
    if args.cfg_scale > 1.0:
        negative_prompt_embeds, negative_prompt_mask = _encoded_prompt(pipe, " ", args.max_sequence_length)
        if negative_prompt_mask is None:
            negative_prompt_mask = torch.ones(
                negative_prompt_embeds.shape[:2], dtype=torch.long, device=device
            )
    if args.precision == "bf16":
        pipe.register_modules(text_encoder=None)
        gc.collect()
        pipe.enable_sequential_cpu_offload(device="cuda")
        device = pipe._execution_device
        prompt_embeds = prompt_embeds.to(device)
        prompt_mask = prompt_mask.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_prompt_mask = negative_prompt_mask.to(device)

    specs: List[Tuple[str, int]] = [(method, step) for method in args.methods for step in args.merge_steps]
    batch_size = 2 + len(specs)
    latent_channels = pipe.transformer.config.in_channels // 4
    seed_latents = []
    for seed in (args.seed_a, args.seed_b):
        seed_latents.append(
            pipe.prepare_latents(
                1,
                latent_channels,
                args.height,
                args.width,
                prompt_embeds.dtype,
                device,
                torch.Generator(device=device).manual_seed(seed),
            )
        )
    initial_latents = torch.cat(seed_latents + [seed_latents[0].clone() for _ in specs], dim=0)
    merge_events: List[Dict[str, object]] = []

    def merge_callback(_pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"].clone()
        for output_index, (method, merge_step) in enumerate(specs, start=2):
            if step == merge_step:
                left = latents[0:1]
                right = latents[1:2]
                merged = merge_latents(left, right, method, args.alpha)
                latents[output_index : output_index + 1] = merged
                merge_events.append(
                    {
                        "output_index": output_index,
                        "method": method,
                        "step": step,
                        "timestep": float(timestep),
                        "merged_sha256": tensor_sha256(merged),
                        "left_stats": tensor_stats(left),
                        "right_stats": tensor_stats(right),
                        "merged_stats": tensor_stats(merged),
                    }
                )
        return {"latents": latents}

    torch.cuda.reset_peak_memory_stats(device)
    result = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds.repeat(batch_size, 1, 1),
        prompt_embeds_mask=prompt_mask.repeat(batch_size, 1),
        negative_prompt_embeds=(
            negative_prompt_embeds.repeat(batch_size, 1, 1)
            if negative_prompt_embeds is not None
            else None
        ),
        negative_prompt_embeds_mask=(
            negative_prompt_mask.repeat(batch_size, 1)
            if negative_prompt_mask is not None
            else None
        ),
        latents=initial_latents,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.cfg_scale,
        max_sequence_length=args.max_sequence_length,
        callback_on_step_end=merge_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    peak_cuda_bytes = torch.cuda.max_memory_allocated(device)

    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root / "results" / f"qwen_latent_merge_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = [f"seed {args.seed_a}", f"seed {args.seed_b}"] + [
        f"{method} alpha={args.alpha:g} step={step}" for method, step in specs
    ]
    image_records = []
    for index, (image, label) in enumerate(zip(result.images, labels)):
        filename = f"{index:02d}.png"
        image_path = output_dir / filename
        image.save(image_path)
        image_records.append(
            {
                "index": index,
                "label": label,
                "file": filename,
                "sha256": file_sha256(image_path),
            }
        )
    grid_path = output_dir / "comparison_grid.png"
    make_grid(result.images, labels, args.grid_columns).save(grid_path)
    manifest = {
        "schema": "sparse_unified_model_qwen_latent_merge_v1",
        "status": "completed",
        "claim_eligible": False,
        "source_commit": _source_commit(repo_root),
        "prompt": args.prompt,
        "prompt_source": (
            {
                "metadata_file": str(Path(args.metadata_file).resolve()),
                "metadata_file_sha256": file_sha256(Path(args.metadata_file)),
                "prompt_index": args.prompt_index,
            }
            if args.metadata_file
            else {"literal_prompt": True}
        ),
        "shared_prompt_embedding_sha256": tensor_sha256(prompt_embeds),
        "model_path": str(Path(args.model_path).resolve()),
        "generation": {
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "initial_latent_a_sha256": tensor_sha256(seed_latents[0]),
            "initial_latent_b_sha256": tensor_sha256(seed_latents[1]),
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "alpha": args.alpha,
            "merge_specs": [{"method": method, "step": step} for method, step in specs],
        },
        "merge_events": merge_events,
        "runtime": {
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "low_memory": args.low_memory,
            "precision": args.precision,
            "vae_slicing": args.low_memory,
            "vae_tiling": args.low_memory,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
        "images": image_records,
        "artifacts": {
            "comparison_grid": "comparison_grid.png",
            "comparison_grid_sha256": file_sha256(grid_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="A red ceramic teapot on a blue table beside three yellow lemons, natural daylight, no text.")
    parser.add_argument("--metadata-file", default="")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seed-a", type=int, default=101)
    parser.add_argument("--seed-b", type=int, default=202)
    parser.add_argument("--methods", type=lambda value: value.split(","), default=["lerp", "slerp"])
    parser.add_argument("--merge-steps", type=lambda value: [int(item) for item in value.split(",")], default=[3])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-weight", default="Qwen-Image-Lightning-8steps-V1.0.safetensors")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    args = parser.parse_args()
    if args.metadata_file:
        rows = [
            json.loads(line)
            for line in Path(args.metadata_file).read_text(encoding="utf-8").splitlines()
        ]
        args.prompt = rows[args.prompt_index]["prompt"]
    if args.seed_a == args.seed_b:
        parser.error("seed A and seed B must differ")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if any(method not in {"lerp", "slerp"} for method in args.methods):
        parser.error("--methods supports only lerp,slerp")
    if any(step < 0 or step >= args.num_inference_steps - 1 for step in args.merge_steps):
        parser.error("merge steps must be before the final denoising step")
    return args


if __name__ == "__main__":
    run(parse_args())
