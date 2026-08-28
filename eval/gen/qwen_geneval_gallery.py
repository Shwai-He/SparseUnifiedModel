#!/usr/bin/env python3
"""Generate a four-sample Qwen-Image gallery in the GenEval directory layout."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re

import torch

from qwen_latent_merge import _encoded_prompt, _load_model, _source_commit, make_grid, tensor_sha256


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


def parse_reasoning_output(text: str) -> tuple[str, str]:
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.IGNORECASE | re.DOTALL)
    prompt_match = re.search(
        r"<generation_prompt>\s*(.*?)\s*</generation_prompt>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if reasoning_match is None or prompt_match is None:
        raise ValueError("Reasoner output must contain reasoning and generation_prompt sections")
    reasoning = reasoning_match.group(1).strip()
    generation_prompt = prompt_match.group(1).strip()
    if not reasoning or not generation_prompt:
        raise ValueError("Reasoning and generation_prompt sections must be non-empty")
    return reasoning, generation_prompt


def reason_about_prompt(pipe, prompt: str, max_new_tokens: int) -> dict[str, str]:
    request = (
        "<|im_start|>system\n"
        + REASONING_INSTRUCTION
        + "<|im_end|>\n<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    device = next(pipe.text_encoder.parameters()).device
    inputs = pipe.tokenizer(request, return_tensors="pt").to(device)
    with torch.no_grad():
        output = pipe.text_encoder.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
    raw_output = pipe.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    try:
        reasoning, generation_prompt = parse_reasoning_output(raw_output)
    except ValueError:
        print(
            json.dumps({"event": "reasoner_parse_failed", "raw_output": raw_output}, sort_keys=True),
            flush=True,
        )
        raise
    return {
        "original_prompt": prompt,
        "raw_output": raw_output,
        "reasoning": reasoning,
        "generation_prompt": generation_prompt,
    }


def run(args: argparse.Namespace) -> Path:
    metadata_path = Path(args.metadata_file).resolve()
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata = rows[args.prompt_index]
    prompt = metadata["prompt"]
    seeds = args.seeds or [args.seed + args.prompt_index * args.num_images + i for i in range(args.num_images)]
    args.num_images = len(seeds)

    model = _load_model(args)
    pipe = model.pipe
    if args.low_memory:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    reasoning_record = None
    if args.reason_before_generation:
        reasoning_record = reason_about_prompt(pipe, prompt, args.reasoning_max_new_tokens)
        prompt = reasoning_record["generation_prompt"]
        print(json.dumps(reasoning_record, indent=2, sort_keys=True), flush=True)

    device = pipe._execution_device
    prompt_embeds, prompt_mask = _encoded_prompt(pipe, prompt, args.max_sequence_length)
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

    latent_channels = pipe.transformer.config.in_channels // 4
    seed_latents = [
        pipe.prepare_latents(
            1,
            latent_channels,
            args.height,
            args.width,
            prompt_embeds.dtype,
            device,
            torch.Generator(device=device).manual_seed(seed),
        )
        for seed in seeds
    ]
    torch.cuda.reset_peak_memory_stats(device)
    pipeline_kwargs = {
        "prompt": None,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_mask,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.cfg_scale,
        "max_sequence_length": args.max_sequence_length,
    }
    if negative_prompt_embeds is not None:
        pipeline_kwargs.update(
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_mask,
        )
    if args.generation_mode == "batched":
        batched_kwargs = {
            **pipeline_kwargs,
            "prompt_embeds": prompt_embeds.repeat(args.num_images, 1, 1),
            "prompt_embeds_mask": prompt_mask.repeat(args.num_images, 1),
            "latents": torch.cat(seed_latents, dim=0),
        }
        if negative_prompt_embeds is not None:
            batched_kwargs.update(
                negative_prompt_embeds=negative_prompt_embeds.repeat(args.num_images, 1, 1),
                negative_prompt_embeds_mask=negative_prompt_mask.repeat(args.num_images, 1),
            )
        result_images = pipe(
            **batched_kwargs
        ).images
    else:
        result_images = [
            pipe(**{**pipeline_kwargs, "latents": latent}).images[0]
            for latent in seed_latents
        ]
    peak_cuda_bytes = torch.cuda.max_memory_allocated(device)

    output_dir = Path(args.output_dir).resolve()
    prompt_dir = output_dir / f"{args.prompt_index:05d}"
    sample_dir = prompt_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=False)
    (prompt_dir / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    image_records = []
    for image_index, (seed, image) in enumerate(zip(seeds, result_images)):
        image_path = sample_dir / f"{image_index:05d}.png"
        image.save(image_path)
        image_records.append(
            {"index": image_index, "seed": seed, "file": str(image_path.relative_to(output_dir)), "sha256": file_sha256(image_path)}
        )
    gallery_path = output_dir / "gallery.png"
    make_grid(result_images, [f"seed {seed}" for seed in seeds], args.num_images).save(gallery_path)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema": (
            "sparse_unified_model_qwen_reasoned_gallery_v1"
            if args.reason_before_generation
            else "sparse_unified_model_qwen_geneval_gallery_v1"
        ),
        "status": "completed",
        "claim_eligible": False,
        "source_commit": _source_commit(repo_root),
        "metadata_file": str(metadata_path),
        "metadata_file_sha256": file_sha256(metadata_path),
        "prompt_index": args.prompt_index,
        "metadata": metadata,
        "reasoning": reasoning_record,
        "reasoning_instruction_sha256": (
            hashlib.sha256(REASONING_INSTRUCTION.encode()).hexdigest() if reasoning_record is not None else None
        ),
        "shared_prompt_embedding_sha256": tensor_sha256(prompt_embeds),
        "generation": {
            "base_seed": args.seed,
            "seeds": seeds,
            "generation_mode": args.generation_mode,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "initial_latent_sha256": [tensor_sha256(latent) for latent in seed_latents],
        },
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
        "artifacts": {"gallery": "gallery.png", "gallery_sha256": file_sha256(gallery_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-index", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--generation-mode", choices=["sequential", "batched"], default="sequential")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--lora-weight", default="Qwen-Image-Lightning-8steps-V1.0.safetensors")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--reason-before-generation", action="store_true")
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.prompt_index < 0 or args.num_images < 1 or args.reasoning_max_new_tokens < 1:
        parser.error("prompt index must be non-negative and generation counts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
