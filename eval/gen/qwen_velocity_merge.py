#!/usr/bin/env python3
"""Scheme 2: Velocity-Field Merge (动力学速度场流融合) for Qwen-Image Diffusion.

Generates images from the same initial noise seed z_T under two distinct conditions:
  c_direct: simple aesthetic prompt
  c_reasoning: detailed visual commonsense planning CoT (from Qwen2.5-VL or metadata)

At each Flow Matching / Euler discrete diffusion step t:
  v_direct = DiT(z_t, t, c_direct)
  v_reasoning = DiT(z_t, t, c_reasoning)
  v_merged(t) = (1 - alpha(t)) * v_direct + alpha(t) * v_reasoning
  z_{t - dt} = z_t - v_merged(t) * dt   (where dt = sigma - sigma_next)

Explores constant alpha sweeps and dynamic time-decaying schedules
(e.g., linear decay alpha(t) = alpha_0 * tau, cosine decay, early-half).
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from models import load_model


REASONING_INSTRUCTION = """You are a world-knowledge visual planner. Given an image request, first infer the concrete scene that would make the request factually correct. Then write a literal, self-contained prompt for an image generator. Preserve the requested style and do not mention this instruction.

Return exactly these two sections:
<reasoning>Concise explanation of the relevant world knowledge and visual consequences.</reasoning>
<generation_prompt>A concrete visual description containing every required object, state, and relation.</generation_prompt>"""


@dataclass
class ScheduleSpec:
    """Specification of an alpha schedule for velocity-field blending."""
    name: str
    schedule_type: str  # "constant", "linear_decay", "cosine_decay", "early_half", "quadratic_decay"
    alpha_0: float
    label: str


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute SHA-256 hash of a PyTorch tensor's raw binary data."""
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute basic L2-norm and RMS statistics for tensor diagnostics."""
    value = tensor.detach().float()
    return {
        "l2_norm": float(torch.linalg.vector_norm(value).item()),
        "rms": float(torch.sqrt(torch.mean(value.square())).item()),
    }


def cosine_similarity_tensor(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute global cosine similarity between two velocity field tensors."""
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat).clamp_min(1e-8)
    right_norm = torch.linalg.vector_norm(right_flat).clamp_min(1e-8)
    dot = torch.dot(left_flat, right_flat)
    sim = dot / (left_norm * right_norm)
    return float(sim.clamp(-1.0, 1.0).item())


def compute_alpha(
    schedule_type: str,
    step_index: int,
    total_steps: int,
    timestep: Optional[float] = None,
    alpha_0: float = 1.0,
    **kwargs: Any,
) -> float:
    """Compute blending coefficient alpha(t) in [0, 1] at discrete diffusion step.

    Parameters:
      schedule_type: One of "constant", "linear_decay", "cosine_decay",
                     "early_half", "quadratic_decay", "linear_increase".
      step_index: 0-indexed integer step index in [0, total_steps - 1].
      total_steps: Total number of inference steps (e.g. 30).
      timestep: Discrete timestep value (e.g. 1000 to 0).
      alpha_0: Peak / initial weight for reasoning velocity in [0.0, 1.0].

    Returns:
      Scalar alpha in [0.0, 1.0].
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if not (0.0 <= alpha_0 <= 1.0):
        raise ValueError(f"alpha_0 must be in [0, 1], got {alpha_0}")
    if not (0 <= step_index < total_steps):
        raise ValueError(f"step_index {step_index} out of bounds [0, {total_steps - 1}]")

    # Normalized time tau in [0, 1]: tau = 1 at step 0 (coarse layout / pure noise),
    # and tau = 0 at step total_steps - 1 (fine detail / clean image).
    tau = 1.0 - (float(step_index) / float(total_steps - 1)) if total_steps > 1 else 1.0

    if schedule_type == "constant":
        alpha = alpha_0
    elif schedule_type == "linear_decay":
        # Linear decay: alpha(t) = alpha_0 * tau
        alpha = alpha_0 * tau
    elif schedule_type == "cosine_decay":
        # Smooth cosine S-curve decay from alpha_0 down to 0.0
        alpha = alpha_0 * 0.5 * (1.0 + math.cos(math.pi * (1.0 - tau)))
    elif schedule_type == "early_half":
        # Reasoning active during first half of steps, direct only during second half
        alpha = alpha_0 if step_index < (total_steps // 2) else 0.0
    elif schedule_type == "quadratic_decay":
        alpha = alpha_0 * (tau ** 2)
    elif schedule_type == "linear_increase":
        # Ablation control: direct early, reasoning late
        alpha = alpha_0 * (1.0 - tau)
    else:
        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    return float(max(0.0, min(1.0, alpha)))


def merge_velocities(
    v_direct: torch.Tensor,
    v_reasoning: torch.Tensor,
    alpha: float,
    method: str = "linear",
) -> torch.Tensor:
    """Merge direct and reasoning velocity fields with coefficient alpha.

    Parameters:
      v_direct: Velocity field under direct prompt c_direct.
      v_reasoning: Velocity field under reasoning prompt c_reasoning.
      alpha: Blending weight in [0.0, 1.0].
      method: Merge strategy ("linear", "norm_preserving", "slerp").

    Returns:
      v_merged = (1 - alpha) * v_direct + alpha * v_reasoning (or scaled variant).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if v_direct.shape != v_reasoning.shape:
        raise ValueError(
            f"Shape mismatch: v_direct {v_direct.shape} vs v_reasoning {v_reasoning.shape}"
        )

    if alpha == 0.0:
        return v_direct
    if alpha == 1.0:
        return v_reasoning

    if method == "linear":
        return (1.0 - alpha) * v_direct + alpha * v_reasoning

    elif method == "norm_preserving":
        # Interpolate linearly, then rescale to convex combination of component norms
        v_lin = (1.0 - alpha) * v_direct.float() + alpha * v_reasoning.float()
        norm_direct = torch.linalg.vector_norm(v_direct.float(), dim=-1, keepdim=True)
        norm_reasoning = torch.linalg.vector_norm(v_reasoning.float(), dim=-1, keepdim=True)
        target_norm = (1.0 - alpha) * norm_direct + alpha * norm_reasoning
        lin_norm = torch.linalg.vector_norm(v_lin, dim=-1, keepdim=True).clamp_min(1e-8)
        v_scaled = v_lin * (target_norm / lin_norm)
        return v_scaled.to(dtype=v_direct.dtype)

    elif method == "slerp":
        from qwen_latent_merge import slerp
        return slerp(v_direct, v_reasoning, alpha)

    else:
        raise ValueError(f"Unsupported merge method: {method}")


def apply_qwen_cfg(
    cond_pred: torch.Tensor,
    uncond_pred: Optional[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    """Apply Qwen-Image style true Classifier-Free Guidance and norm rescaling."""
    if cfg_scale <= 1.0 or uncond_pred is None:
        return cond_pred
    comb_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).clamp_min(1e-8)
    return comb_pred * (cond_norm / noise_norm)


def euler_step_flow_match(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Discrete Flow Matching Euler step: z_{next} = z + dt * velocity.

    Note: In FlowMatchEulerDiscreteScheduler, sigmas decrease from 1 to 0,
    so dt = sigma_next - sigma < 0.
    """
    return sample.to(torch.float32) + dt * velocity.to(torch.float32)


def make_grid(images: Sequence[Image.Image], labels: Sequence[str], columns: int) -> Image.Image:
    """Arrange images and textual headers into a comparison grid."""
    font = ImageFont.load_default()
    label_height = 32
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 8, y + 8), label, fill="black", font=font)
        canvas.paste(ImageOps.pad(image.convert("RGB"), (width, height)), (x, y + label_height))
    return canvas


def _encoded_prompt(pipe: Any, prompt: str, max_sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompt using Qwen-Image pipeline's text encoder."""
    encoded = pipe.encode_prompt(
        prompt=[prompt],
        device=pipe._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return encoded[0], encoded[1]


def _source_commit(repo_root: Path) -> str:
    """Retrieve the current 40-character Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown_commit"


def parse_reasoning_output(text: str) -> Tuple[str, str]:
    """Parse reasoning and generation_prompt sections from Qwen2.5-VL output."""
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, flags=re.IGNORECASE | re.DOTALL)
    prompt_match = re.search(
        r"<generation_prompt>\s*(.*?)\s*</generation_prompt>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if reasoning_match is None or prompt_match is None:
        raise ValueError("Reasoner output must contain <reasoning> and <generation_prompt> sections")
    reasoning = reasoning_match.group(1).strip()
    generation_prompt = prompt_match.group(1).strip()
    if not reasoning or not generation_prompt:
        raise ValueError("Reasoning and generation_prompt sections must be non-empty")
    return reasoning, generation_prompt


def reason_about_prompt(pipe: Any, prompt: str, max_new_tokens: int) -> Dict[str, str]:
    """Run Qwen2.5-VL language model to generate visual commonsense reasoning CoT."""
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
    reasoning, generation_prompt = parse_reasoning_output(raw_output)
    return {
        "original_prompt": prompt,
        "raw_output": raw_output,
        "reasoning": reasoning,
        "generation_prompt": generation_prompt,
    }


def _load_model(args: argparse.Namespace) -> Any:
    """Load Qwen-Image diffusion model with NF4 or BF16 precision."""
    pipeline_kwargs: Dict[str, Any] = {}
    text_encoder_kwargs: Dict[str, Any] = {}
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
        raise ValueError(f"Unsupported precision: {precision}")

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


def decode_latents(pipe: Any, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    """Unpack DiT latents, apply VAE standard scaling, decode and return PIL Image."""
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = unpacked.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(unpacked.device, unpacked.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(unpacked.device, unpacked.dtype)
    unpacked = unpacked / latents_std + latents_mean
    image_tensor = pipe.vae.decode(unpacked, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]


def run_velocity_merge_trajectory(
    pipe: Any,
    initial_latents: torch.Tensor,
    spec: ScheduleSpec,
    prompt_embeds_d: torch.Tensor,
    prompt_mask_d: torch.Tensor,
    prompt_embeds_r: torch.Tensor,
    prompt_mask_r: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_mask: Optional[torch.Tensor],
    height: int,
    width: int,
    num_inference_steps: int,
    cfg_scale: float,
    merge_method: str = "linear",
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Execute a single diffusion trajectory with velocity field merge.

    Starts from the exact frozen initial_latents z_T and computes blended
    velocities at every discrete Euler step.
    """
    device = pipe._execution_device
    latents = initial_latents.clone().to(device)
    batch_size = latents.shape[0]

    # Configure scheduler timesteps
    import numpy as np

    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]

    # Calculate mu shift for dynamic flow matching
    base_seq_len = pipe.scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = pipe.scheduler.config.get("max_image_seq_len", 4096)
    base_shift = pipe.scheduler.config.get("base_shift", 0.5)
    max_shift = pipe.scheduler.config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    pipe.scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    timesteps = pipe.scheduler.timesteps
    pipe.scheduler.set_begin_index(0)

    img_shapes = [[(1, height // pipe.vae_scale_factor // 2, width // pipe.vae_scale_factor // 2)]] * batch_size

    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], 1.0, device=device, dtype=torch.float32).expand(batch_size)
    else:
        guidance = None

    attention_kwargs = pipe.attention_kwargs or {}
    txt_seq_lens_d = prompt_mask_d.sum(dim=1).tolist() if prompt_mask_d is not None else None
    txt_seq_lens_r = prompt_mask_r.sum(dim=1).tolist() if prompt_mask_r is not None else None
    neg_txt_seq_lens = negative_prompt_mask.sum(dim=1).tolist() if negative_prompt_mask is not None else None

    step_telemetry: List[Dict[str, Any]] = []

    for step_idx, t in enumerate(timesteps):
        timestep = t.expand(batch_size).to(latents.dtype)
        alpha = compute_alpha(
            spec.schedule_type,
            step_idx,
            num_inference_steps,
            timestep=float(t),
            alpha_0=spec.alpha_0,
        )

        # 1. Evaluate unconditional prediction (shared across both conditions if CFG > 1)
        neg_noise_pred = None
        if cfg_scale > 1.0 and negative_prompt_embeds is not None:
            with pipe.transformer.cache_context("uncond"):
                neg_noise_pred = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=neg_txt_seq_lens,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

        # 2. Evaluate direct velocity
        v_direct = None
        if alpha < 1.0:
            with pipe.transformer.cache_context("cond"):
                raw_direct = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_mask_d,
                    encoder_hidden_states=prompt_embeds_d,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens_d,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            v_direct = apply_qwen_cfg(raw_direct, neg_noise_pred, cfg_scale)

        # 3. Evaluate reasoning velocity
        v_reasoning = None
        if alpha > 0.0:
            with pipe.transformer.cache_context("cond"):
                raw_reasoning = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_mask_r,
                    encoder_hidden_states=prompt_embeds_r,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens_r,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            v_reasoning = apply_qwen_cfg(raw_reasoning, neg_noise_pred, cfg_scale)

        # 4. Merge velocities
        if alpha == 0.0:
            v_merged = v_direct
            cos_sim = 1.0
        elif alpha == 1.0:
            v_merged = v_reasoning
            cos_sim = 1.0
        else:
            cos_sim = cosine_similarity_tensor(v_direct, v_reasoning)
            v_merged = merge_velocities(v_direct, v_reasoning, alpha, method=merge_method)

        # 5. Advance latent along merged velocity field
        latents = pipe.scheduler.step(v_merged, t, latents, return_dict=False)[0]

        # 6. Record step telemetry
        metric_record: Dict[str, Any] = {
            "schedule": spec.name,
            "step": step_idx,
            "timestep": float(t.item() if isinstance(t, torch.Tensor) else t),
            "alpha": float(alpha),
            "cosine_similarity": float(cos_sim),
            "v_merged_stats": tensor_stats(v_merged),
            "latent_stats": tensor_stats(latents),
        }
        if v_direct is not None:
            metric_record["v_direct_stats"] = tensor_stats(v_direct)
        if v_reasoning is not None:
            metric_record["v_reasoning_stats"] = tensor_stats(v_reasoning)

        step_telemetry.append(metric_record)

    image = decode_latents(pipe, latents, height, width)
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()

    return image, step_telemetry


def build_schedule_specs(
    alphas: Sequence[float],
    schedules: Sequence[str],
    schedule_alpha_0: float = 1.0,
) -> List[ScheduleSpec]:
    """Build list of ScheduleSpec objects covering constant alpha sweep and dynamic schedules."""
    specs: List[ScheduleSpec] = []

    # Constant alphas
    for a in alphas:
        label = f"alpha={a:g}"
        if a == 0.0:
            label += " (Direct)"
        elif a == 1.0:
            label += " (Reasoning)"
        specs.append(
            ScheduleSpec(
                name=f"const_alpha_{a:g}",
                schedule_type="constant",
                alpha_0=float(a),
                label=label,
            )
        )

    # Dynamic decay schedules
    for s in schedules:
        clean_name = s.strip()
        if not clean_name:
            continue
        specs.append(
            ScheduleSpec(
                name=clean_name,
                schedule_type=clean_name,
                alpha_0=float(schedule_alpha_0),
                label=f"{clean_name} (a0={schedule_alpha_0:g})",
            )
        )

    return specs


def run(args: argparse.Namespace) -> Path:
    """Main execution flow for Scheme 2: Velocity-Field Merge."""
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Resolve prompts
    prompt_direct = args.prompt
    metadata_record = None
    if args.metadata_file:
        metadata_path = Path(args.metadata_file).resolve()
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        metadata_record = rows[args.prompt_index]
        prompt_direct = metadata_record["prompt"]

    model = _load_model(args)
    pipe = model.pipe
    if args.low_memory:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    device = pipe._execution_device

    # Resolve reasoning prompt
    reasoning_record = None
    if args.reason_before_generation:
        reasoning_record = reason_about_prompt(pipe, prompt_direct, args.reasoning_max_new_tokens)
        prompt_reasoning = reasoning_record["generation_prompt"]
        print(f"[Reasoning CoT Generated]:\n{json.dumps(reasoning_record, indent=2)}")
    elif args.reasoning_prompt:
        prompt_reasoning = args.reasoning_prompt
    elif metadata_record and "explanation" in metadata_record and args.use_metadata_explanation:
        prompt_reasoning = f"{prompt_direct}. Visual requirements: {metadata_record['explanation']}"
    else:
        # Fallback: if no reasoner is requested, use prompt_direct as reasoning
        prompt_reasoning = prompt_direct

    print(f"=== Condition Direct ===\n{prompt_direct}")
    print(f"=== Condition Reasoning ===\n{prompt_reasoning}")

    # Encode prompt conditions
    prompt_embeds_d, prompt_mask_d = _encoded_prompt(pipe, prompt_direct, args.max_sequence_length)
    if prompt_mask_d is None:
        prompt_mask_d = torch.ones(prompt_embeds_d.shape[:2], dtype=torch.long, device=device)

    prompt_embeds_r, prompt_mask_r = _encoded_prompt(pipe, prompt_reasoning, args.max_sequence_length)
    if prompt_mask_r is None:
        prompt_mask_r = torch.ones(prompt_embeds_r.shape[:2], dtype=torch.long, device=device)

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
        prompt_embeds_d = prompt_embeds_d.to(device)
        prompt_mask_d = prompt_mask_d.to(device)
        prompt_embeds_r = prompt_embeds_r.to(device)
        prompt_mask_r = prompt_mask_r.to(device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_prompt_mask = negative_prompt_mask.to(device)

    # Prepare frozen initial latent seed z_T
    latent_channels = pipe.transformer.config.in_channels // 4
    initial_latents = pipe.prepare_latents(
        1,
        latent_channels,
        args.height,
        args.width,
        prompt_embeds_d.dtype,
        device,
        torch.Generator(device=device).manual_seed(args.seed),
    )
    initial_latent_sha = tensor_sha256(initial_latents)

    # Build schedules
    specs = build_schedule_specs(args.alphas, args.schedules, args.schedule_alpha_0)
    print(f"Executing Velocity-Field Merge across {len(specs)} schedules for Seed {args.seed}:")
    for s in specs:
        print(f"  - {s.name}: type={s.schedule_type}, alpha_0={s.alpha_0:g}, label={s.label}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_root / "results" / f"qwen_velocity_merge_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_images: List[Image.Image] = []
    image_records: List[Dict[str, Any]] = []
    all_telemetry: List[Dict[str, Any]] = []

    torch.cuda.reset_peak_memory_stats(device)

    for index, spec in enumerate(specs):
        print(f"\n[{index + 1}/{len(specs)}] Running schedule: {spec.name} ({spec.label})...")
        image, telemetry = run_velocity_merge_trajectory(
            pipe=pipe,
            initial_latents=initial_latents,
            spec=spec,
            prompt_embeds_d=prompt_embeds_d,
            prompt_mask_d=prompt_mask_d,
            prompt_embeds_r=prompt_embeds_r,
            prompt_mask_r=prompt_mask_r,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            merge_method=args.merge_method,
        )

        filename = f"{index:02d}_{spec.name}.png"
        image_path = output_dir / filename
        image.save(image_path)
        generated_images.append(image)

        image_records.append(
            {
                "index": index,
                "schedule_name": spec.name,
                "schedule_type": spec.schedule_type,
                "alpha_0": spec.alpha_0,
                "label": spec.label,
                "file": filename,
                "sha256": file_sha256(image_path),
            }
        )
        all_telemetry.extend(telemetry)

    peak_cuda_bytes = torch.cuda.max_memory_allocated(device)

    # Create comparison grid
    grid_path = output_dir / "comparison_grid.png"
    grid_image = make_grid(generated_images, [s.label for s in specs], args.grid_columns)
    grid_image.save(grid_path)

    # Save detailed step-by-step trajectory metrics JSONL
    trajectory_jsonl_path = output_dir / "velocity_trajectory.jsonl"
    with trajectory_jsonl_path.open("w", encoding="utf-8") as f:
        for record in all_telemetry:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # Build manifest
    manifest = {
        "schema": "sparse_unified_model_qwen_velocity_merge_v1",
        "status": "completed",
        "claim_eligible": False,
        "source_commit": _source_commit(repo_root),
        "prompt_direct": prompt_direct,
        "prompt_reasoning": prompt_reasoning,
        "prompt_source": (
            {
                "metadata_file": str(Path(args.metadata_file).resolve()),
                "metadata_file_sha256": file_sha256(Path(args.metadata_file)),
                "prompt_index": args.prompt_index,
                "metadata_record": metadata_record,
            }
            if args.metadata_file
            else {"literal_prompt": True}
        ),
        "reasoning_record": reasoning_record,
        "generation": {
            "seed": args.seed,
            "initial_latent_sha256": initial_latent_sha,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "merge_method": args.merge_method,
            "precision": args.precision,
            "low_memory": args.low_memory,
            "schedules": [
                {
                    "name": s.name,
                    "schedule_type": s.schedule_type,
                    "alpha_0": s.alpha_0,
                    "label": s.label,
                }
                for s in specs
            ],
        },
        "images": image_records,
        "artifacts": {
            "comparison_grid": "comparison_grid.png",
            "comparison_grid_sha256": file_sha256(grid_path),
            "velocity_trajectory": "velocity_trajectory.jsonl",
            "velocity_trajectory_sha256": file_sha256(trajectory_jsonl_path),
        },
        "runtime": {
            "torch": torch.__version__,
            "diffusers": __import__("diffusers").__version__,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n[Velocity Merge Completed Successfully]")
    print(f"Output Directory: {output_dir}")
    print(f"Comparison Grid: {grid_path}")
    print(f"Manifest: {manifest_path}")

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", required=True, help="Path to Qwen-Image model checkpoint")
    parser.add_argument(
        "--prompt",
        default="A red ceramic teapot on a blue table beside three yellow lemons, natural daylight, no text.",
        help="Direct prompt (c_direct)",
    )
    parser.add_argument(
        "--reasoning-prompt",
        default="",
        help="Explicit visual reasoning prompt (c_reasoning). If empty, can be generated or read from metadata.",
    )
    parser.add_argument(
        "--reason-before-generation",
        action="store_true",
        help="Use Qwen2.5-VL to autonomously generate commonsense visual planning CoT",
    )
    parser.add_argument(
        "--reasoning-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens for Qwen2.5-VL reasoner",
    )
    parser.add_argument("--metadata-file", default="", help="Path to JSONL prompt registry (e.g. WISE)")
    parser.add_argument("--prompt-index", type=int, default=0, help="Index of prompt in metadata file")
    parser.add_argument(
        "--use-metadata-explanation",
        action="store_true",
        help="Use WISE explanation field from metadata file as reasoning reference if present",
    )
    parser.add_argument("--seed", type=int, default=101, help="Fixed initial noise seed z_T")
    parser.add_argument(
        "--alphas",
        type=lambda v: [float(item) for item in v.split(",") if item.strip()],
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Comma-separated constant alpha values to evaluate",
    )
    parser.add_argument(
        "--schedules",
        type=lambda v: [item.strip() for item in v.split(",") if item.strip()],
        default=["linear_decay", "cosine_decay"],
        help="Comma-separated dynamic schedules (linear_decay, cosine_decay, early_half, quadratic_decay)",
    )
    parser.add_argument(
        "--schedule-alpha-0",
        type=float,
        default=1.0,
        help="Base alpha_0 for dynamic schedules (e.g. 1.0)",
    )
    parser.add_argument(
        "--merge-method",
        choices=["linear", "norm_preserving", "slerp"],
        default="linear",
        help="Velocity merge formula (linear, norm_preserving, or slerp)",
    )
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of Euler denoising steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--max-sequence-length", type=int, default=256, help="Max sequence length for text encoder")
    parser.add_argument("--grid-columns", type=int, default=4, help="Number of columns in output grid")
    parser.add_argument("--lora-path", default="", help="Optional LoRA path")
    parser.add_argument("--lora-weight", default="Qwen-Image-Lightning-8steps-V1.0.safetensors", help="LoRA weight filename")
    parser.add_argument("--output-dir", default="", help="Directory to save generated outputs and manifest")
    parser.add_argument("--low-memory", action="store_true", help="Enable VAE slicing and tiling")
    parser.add_argument("--precision", choices=["nf4", "bf16"], default="nf4", help="Inference precision")

    args = parser.parse_args()

    for a in args.alphas:
        if not (0.0 <= a <= 1.0):
            parser.error(f"Alpha value {a} out of range [0, 1]")

    if not (0.0 <= args.schedule_alpha_0 <= 1.0):
        parser.error(f"--schedule-alpha-0 {args.schedule_alpha_0} out of range [0, 1]")

    if args.num_inference_steps < 2:
        parser.error("--num-inference-steps must be at least 2")

    return args


if __name__ == "__main__":
    run(parse_args())
