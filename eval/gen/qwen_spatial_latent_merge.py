#!/usr/bin/env python3
"""Spatial-Selective Latent Merge for Qwen-Image DiT (Scheme 1: Cross-Seed Complementary Fusion).

Performs spatial mask grafting during early-to-mid diffusion denoising steps:
    z_k^{merged} = M * z_k^B + (1 - M) * z_k^A
where M is a spatial mask (e.g., bounding box with Gaussian boundary falloff).
The remaining (N - k) denoising steps continue natively on z_k^{merged}, acting
as natural latent inpainting to harmonize lighting, shadows, and seam boundaries.
"""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

from models import load_model


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 digest of a PyTorch tensor."""
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute summary L2 norm and RMS of a tensor."""
    value = tensor.detach().float()
    return {
        "l2_norm": torch.linalg.vector_norm(value).item(),
        "rms": torch.sqrt(torch.mean(value.square())).item(),
    }


def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pack 5D VAE latent tensor [B, C, 1, H, W] into 3D DiT sequence tokens [B, S, 4*C]."""
    if latents.ndim == 5:
        latents = latents.squeeze(2)
    # latents is now [B, num_channels_latents, H, W]
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Unpack 3D DiT sequence tokens [B, S, C] into 5D VAE latents [B, C // 4, 1, H, W]."""
    batch_size, num_patches, channels = latents.shape
    # Support both latent dimensions (e.g. 64x64) and pixel dimensions (e.g. 512x512)
    if (height // 2) * (width // 2) == num_patches:
        h_lat = height
        w_lat = width
    else:
        h_lat = 2 * (int(height) // (vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (vae_scale_factor * 2))

    if (h_lat // 2) * (w_lat // 2) != num_patches:
        raise ValueError(
            f"Latent patch mismatch: (h_lat//2)*(w_lat//2) = {(h_lat//2)*(w_lat//2)} "
            f"does not match num_patches={num_patches} for height={height}, width={width}"
        )

    latents = latents.view(batch_size, h_lat // 2, w_lat // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, h_lat, w_lat)
    return latents


def create_gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Generate 1D normalized Gaussian kernel for spatial boundary smoothing."""
    if sigma <= 0.0:
        return torch.tensor([1.0], dtype=torch.float32)
    radius = int(math.ceil(truncate * sigma))
    radius = max(radius, 1)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def apply_gaussian_smoothing(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable 2D Gaussian smoothing to a 4D mask tensor [B, 1, H, W]."""
    if sigma <= 0.0:
        return mask.clamp(0.0, 1.0)
    kernel_1d = create_gaussian_kernel1d(sigma)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, *kernel_2d.shape).to(
        device=mask.device, dtype=mask.dtype
    )
    pad = kernel_1d.shape[0] // 2
    padded = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    smoothed = F.conv2d(padded, kernel_2d)
    return smoothed.clamp(0.0, 1.0)


def build_spatial_mask(
    height_lat: int,
    width_lat: int,
    mask_type: str = "box",
    box: Optional[Sequence[float]] = None,
    center: Optional[Sequence[float]] = None,
    radius: Optional[Sequence[float]] = None,
    split_ratio: float = 0.5,
    split_axis: str = "vertical",
    sigma: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a normalized 2D spatial mask [1, 1, height_lat, width_lat] with values in [0, 1].

    Args:
        height_lat: Latent grid height.
        width_lat: Latent grid width.
        mask_type: 'box', 'gaussian', 'split', or 'full'.
        box: Normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0].
        center: Normalized [cy, cx] in [0.0, 1.0] for Gaussian ellipsoids.
        radius: Normalized [ry, rx] in [0.0, 1.0] for Gaussian ellipsoids.
        split_ratio: Split line location in [0.0, 1.0] for 'split' mask.
        split_axis: 'vertical' (left/right) or 'horizontal' (top/bottom).
        sigma: Standard deviation for Gaussian boundary smoothing (in latent pixels).
        device: PyTorch device.
        dtype: PyTorch floating-point data type.

    Returns:
        Tensor of shape [1, 1, height_lat, width_lat] strictly in [0.0, 1.0].
    """
    if height_lat <= 0 or width_lat <= 0:
        raise ValueError(f"Latent dimensions must be positive, got ({height_lat}, {width_lat})")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    y = torch.linspace(0.0, 1.0, height_lat, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width_lat, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    mask_2d = torch.zeros((height_lat, width_lat), device=device, dtype=torch.float32)

    if mask_type == "box":
        if box is None:
            box = [0.0, 0.5, 1.0, 1.0]  # default right half
        if len(box) != 4:
            raise ValueError(f"box must contain 4 normalized coordinates [ymin, xmin, ymax, xmax], got {box}")
        ymin, xmin, ymax, xmax = [float(v) for v in box]
        if not (0.0 <= ymin <= ymax <= 1.0 and 0.0 <= xmin <= xmax <= 1.0):
            raise ValueError(f"box coordinates must satisfy 0 <= ymin <= ymax <= 1 and 0 <= xmin <= xmax <= 1, got {box}")
        inside = (grid_y >= ymin) & (grid_y <= ymax) & (grid_x >= xmin) & (grid_x <= xmax)
        mask_2d[inside] = 1.0

    elif mask_type == "gaussian":
        if center is None:
            center = [0.5, 0.75]
        if radius is None:
            radius = [0.25, 0.25]
        cy, cx = float(center[0]), float(center[1])
        ry, rx = max(float(radius[0]), 1e-4), max(float(radius[1]), 1e-4)
        dist_sq = ((grid_y - cy) / ry) ** 2 + ((grid_x - cx) / rx) ** 2
        mask_2d = torch.exp(-0.5 * dist_sq)

    elif mask_type == "split":
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError(f"split_ratio must be in [0, 1], got {split_ratio}")
        if split_axis == "vertical":
            mask_2d[grid_x >= split_ratio] = 1.0
        elif split_axis == "horizontal":
            mask_2d[grid_y >= split_ratio] = 1.0
        else:
            raise ValueError(f"unsupported split_axis: {split_axis}, must be 'vertical' or 'horizontal'")

    elif mask_type == "full":
        mask_2d.fill_(1.0)
    else:
        raise ValueError(f"unsupported mask_type: {mask_type}, choices: box, gaussian, split, full")

    mask_4d = mask_2d.view(1, 1, height_lat, width_lat)
    if sigma > 0.0:
        mask_4d = apply_gaussian_smoothing(mask_4d, sigma)
    return mask_4d.to(dtype=dtype)


def mask_to_patch_tokens(
    mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Downsample latent 4D mask [1, 1, H_lat, W_lat] to DiT token sequence [1, S, 1]."""
    # 2x2 average pooling from VAE latent cells to DiT patch tokens
    patch_mask = F.avg_pool2d(mask_4d.float(), kernel_size=2, stride=2)
    batch_size, _, hp, wp = patch_mask.shape
    tokens = patch_mask.permute(0, 2, 3, 1).reshape(batch_size, hp * wp, 1)
    return tokens.to(dtype=mask_4d.dtype)


def spatial_lerp(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Linear spatial interpolation: (1 - alpha * M) * left + (alpha * M) * right."""
    effective_mask = (mask * alpha).clamp(0.0, 1.0)
    return (1.0 - effective_mask) * left + effective_mask * right


def spatial_slerp(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Spherical linear interpolation along the channel dimension weighted by spatial mask.

    Preserves vector norm across the boundary transition zone to prevent norm collapse.
    """
    effective_mask = (mask * alpha).clamp(0.0, 1.0).float()
    left_f = left.float()
    right_f = right.float()

    left_norm = torch.linalg.vector_norm(left_f, dim=channel_dim, keepdim=True).clamp_min(1e-8)
    right_norm = torch.linalg.vector_norm(right_f, dim=channel_dim, keepdim=True).clamp_min(1e-8)

    cosine = (
        ((left_f / left_norm) * (right_f / right_norm))
        .sum(dim=channel_dim, keepdim=True)
        .clamp(-0.9995, 0.9995)
    )
    angle = torch.acos(cosine)
    denominator = torch.sin(angle).clamp_min(1e-8)

    scale_left = torch.sin((1.0 - effective_mask) * angle) / denominator
    scale_right = torch.sin(effective_mask * angle) / denominator

    merged = scale_left * left_f + scale_right * right_f
    return merged.to(dtype=left.dtype)


def merge_spatial_latents(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
    blend_mode: str = "lerp",
    alpha: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Spatially merge Candidate A (left) and Candidate B (right) with mask M.

    Supports both 5D unpacked VAE latents [B, C, 1, H, W] and 3D packed DiT latents [B, S, C].
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch between left {left.shape} and right {right.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if blend_mode not in {"lerp", "slerp"}:
        raise ValueError(f"unsupported blend_mode: {blend_mode}, choices: lerp, slerp")

    # Case 1: 3D packed DiT latents [B, S, C]
    if left.ndim == 3:
        if height is not None and width is not None:
            # Sub-patch fine precision via unpack -> spatial merge -> repack
            unpacked_left = unpack_latents(left, height, width, vae_scale_factor)
            unpacked_right = unpack_latents(right, height, width, vae_scale_factor)
            # mask should be [1, 1, 1, H_lat, W_lat] or [1, 1, H_lat, W_lat]
            if mask.ndim == 4:
                mask_5d = mask.unsqueeze(2)
            else:
                mask_5d = mask
            mask_5d = mask_5d.to(device=left.device, dtype=left.dtype)
            if blend_mode == "lerp":
                unpacked_merged = spatial_lerp(unpacked_left, unpacked_right, mask_5d, alpha=alpha)
            else:
                unpacked_merged = spatial_slerp(
                    unpacked_left, unpacked_right, mask_5d, alpha=alpha, channel_dim=1
                )
            num_channels_latents = unpacked_merged.shape[1]
            h_lat = unpacked_merged.shape[3]
            w_lat = unpacked_merged.shape[4]
            return pack_latents(unpacked_merged, left.shape[0], num_channels_latents, h_lat, w_lat)
        else:
            # Direct token-level packed merge
            if mask.ndim == 4:
                token_mask = mask_to_patch_tokens(mask)
            else:
                token_mask = mask
            token_mask = token_mask.to(device=left.device, dtype=left.dtype)
            if blend_mode == "lerp":
                return spatial_lerp(left, right, token_mask, alpha=alpha)
            return spatial_slerp(left, right, token_mask, alpha=alpha, channel_dim=-1)

    # Case 2: 5D unpacked latents [B, C, 1, H, W]
    if left.ndim == 5:
        mask_5d = mask if mask.ndim == 5 else mask.unsqueeze(2)
        mask_5d = mask_5d.to(device=left.device, dtype=left.dtype)
        if blend_mode == "lerp":
            return spatial_lerp(left, right, mask_5d, alpha=alpha)
        return spatial_slerp(left, right, mask_5d, alpha=alpha, channel_dim=1)

    # Case 3: 4D unpacked latents [B, C, H, W]
    if left.ndim == 4:
        mask_4d = mask.to(device=left.device, dtype=left.dtype)
        if blend_mode == "lerp":
            return spatial_lerp(left, right, mask_4d, alpha=alpha)
        return spatial_slerp(left, right, mask_4d, alpha=alpha, channel_dim=1)

    raise ValueError(f"unsupported latent tensor dimensionality: {left.ndim}")


def compute_spatial_merge_stats(
    left: torch.Tensor,
    right: torch.Tensor,
    merged: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic statistics and boundary metrics for the merge event."""
    left_f = left.detach().float()
    right_f = right.detach().float()
    merged_f = merged.detach().float()

    l_flat = left_f.reshape(left_f.shape[0], -1)
    r_flat = right_f.reshape(right_f.shape[0], -1)
    m_flat = merged_f.reshape(merged_f.shape[0], -1)

    l_norm = torch.linalg.vector_norm(l_flat, dim=1).item()
    r_norm = torch.linalg.vector_norm(r_flat, dim=1).item()
    m_norm = torch.linalg.vector_norm(m_flat, dim=1).item()

    cos_sim_lr = (
        (l_flat / max(l_norm, 1e-8)) * (r_flat / max(r_norm, 1e-8))
    ).sum(dim=1).item()

    # Mask diagnostics
    mask_flat = mask.detach().float().reshape(1, -1)
    if mask_flat.shape[1] != l_flat.shape[1]:
        # Expand or pool mask to match flat shape
        coverage = (mask_flat > 0.5).float().mean().item()
        cos_sim_in = 1.0
        cos_sim_out = 1.0
    else:
        in_mask = mask_flat > 0.5
        out_mask = mask_flat <= 0.5
        coverage = in_mask.float().mean().item()

        if in_mask.any():
            m_in = m_flat[in_mask]
            r_in = r_flat[in_mask]
            cos_sim_in = (
                (m_in / torch.linalg.vector_norm(m_in).clamp_min(1e-8))
                * (r_in / torch.linalg.vector_norm(r_in).clamp_min(1e-8))
            ).sum().item()
        else:
            cos_sim_in = 1.0

        if out_mask.any():
            m_out = m_flat[out_mask]
            l_out = l_flat[out_mask]
            cos_sim_out = (
                (m_out / torch.linalg.vector_norm(m_out).clamp_min(1e-8))
                * (l_out / torch.linalg.vector_norm(l_out).clamp_min(1e-8))
            ).sum().item()
        else:
            cos_sim_out = 1.0

    return {
        "l2_norm": m_norm,
        "rms": torch.sqrt(torch.mean(merged_f.square())).item(),
        "candidate_a_norm": l_norm,
        "candidate_b_norm": r_norm,
        "cosine_sim_candidate_a_vs_b": cos_sim_lr,
        "cosine_sim_inside_vs_candidate_b": cos_sim_in,
        "cosine_sim_outside_vs_candidate_a": cos_sim_out,
        "mask_coverage_ratio": coverage,
    }


def save_mask_visualization(mask: torch.Tensor, output_path: Path) -> None:
    """Render spatial mask to a PNG image for inspection and auditability."""
    m_np = mask.detach().cpu().float().squeeze().numpy()
    if m_np.ndim > 2:
        m_np = m_np.reshape(m_np.shape[-2], m_np.shape[-1])
    m_scaled = (m_np.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(m_scaled, mode="L")
    image.save(output_path)


def make_grid(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    columns: int,
) -> Image.Image:
    """Construct an annotated image comparison grid."""
    font = ImageFont.load_default()
    label_height = 32
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        draw.text((x + 8, y + 8), label, fill="black", font=font)
        canvas.paste(
            ImageOps.pad(image.convert("RGB"), (width, height)),
            (x, y + label_height),
        )
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
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


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
    """Execute spatial latent merge pipeline and generate audited comparison results."""
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
        negative_prompt_embeds, negative_prompt_mask = _encoded_prompt(
            pipe, " ", args.max_sequence_length
        )
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

    # Calculate latent spatial dimensions
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    h_lat = 2 * (int(args.height) // (vae_scale * 2))
    w_lat = 2 * (int(args.width) // (vae_scale * 2))

    # Construct spatial mask
    box_coords = [float(x) for x in args.mask_box.split(",")] if args.mask_box else None
    center_coords = [float(x) for x in args.mask_center.split(",")] if args.mask_center else None
    radius_coords = [float(x) for x in args.mask_radius.split(",")] if args.mask_radius else None

    spatial_mask = build_spatial_mask(
        height_lat=h_lat,
        width_lat=w_lat,
        mask_type=args.mask_type,
        box=box_coords,
        center=center_coords,
        radius=radius_coords,
        split_ratio=args.split_ratio,
        split_axis=args.split_axis,
        sigma=args.softness_sigma,
        device=device,
        dtype=prompt_embeds.dtype,
    )
    mask_digest = tensor_sha256(spatial_mask)

    # Define merge specs: each (blend_mode, step, sigma)
    specs: List[Tuple[str, int, float]] = [
        (method, step, args.softness_sigma)
        for method in args.blend_modes
        for step in args.merge_steps
    ]
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

    # Initial latents: [Candidate A, Candidate B, Spec 0 clone, Spec 1 clone, ...]
    initial_latents = torch.cat(
        seed_latents + [seed_latents[0].clone() for _ in specs], dim=0
    )
    merge_events: List[Dict[str, Any]] = []

    def merge_callback(_pipe, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"].clone()
        for output_index, (method, merge_step, sigma) in enumerate(specs, start=2):
            if step == merge_step:
                left = latents[0:1]   # Candidate A
                right = latents[1:2]  # Candidate B
                merged = merge_spatial_latents(
                    left=left,
                    right=right,
                    mask=spatial_mask,
                    blend_mode=method,
                    alpha=args.alpha,
                    height=args.height,
                    width=args.width,
                    vae_scale_factor=vae_scale,
                )
                latents[output_index : output_index + 1] = merged
                stats = compute_spatial_merge_stats(left, right, merged, spatial_mask)
                merge_events.append(
                    {
                        "output_index": output_index,
                        "method": method,
                        "step": step,
                        "timestep": float(timestep),
                        "softness_sigma": sigma,
                        "merged_sha256": tensor_sha256(merged),
                        "left_stats": tensor_stats(left),
                        "right_stats": tensor_stats(right),
                        "merged_stats": tensor_stats(merged),
                        "spatial_diagnostics": stats,
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
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_root / "results" / f"qwen_spatial_merge_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save spatial mask visualization
    mask_vis_path = output_dir / "mask_visualization.png"
    save_mask_visualization(spatial_mask, mask_vis_path)

    labels = [f"Candidate A (seed {args.seed_a})", f"Candidate B (seed {args.seed_b})"] + [
        f"{method} step={step} sigma={sigma:g}"
        for method, step, sigma in specs
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
        "schema": "sparse_unified_model_qwen_spatial_latent_merge_v1",
        "scheme": "Scheme 1 (Spatial-Selective Cross-Seed Complementary Fusion)",
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
            "merge_specs": [
                {"method": method, "step": step, "softness_sigma": sigma}
                for method, step, sigma in specs
            ],
        },
        "spatial_mask": {
            "mask_type": args.mask_type,
            "mask_box": box_coords,
            "mask_center": center_coords,
            "mask_radius": radius_coords,
            "split_ratio": args.split_ratio,
            "split_axis": args.split_axis,
            "softness_sigma": args.softness_sigma,
            "mask_sha256": mask_digest,
            "latent_resolution": [h_lat, w_lat],
            "mask_visualization": "mask_visualization.png",
            "mask_visualization_sha256": file_sha256(mask_vis_path),
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
            "mask_visualization": "mask_visualization.png",
            "mask_visualization_sha256": file_sha256(mask_vis_path),
        },
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[SpatialMerge] Outputs successfully staged to: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse and validate command line options for spatial latent merge."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", required=True, help="Path to Qwen-Image model weights.")
    parser.add_argument(
        "--prompt",
        default="A red ceramic teapot on a blue table beside three yellow lemons, natural daylight.",
        help="Text prompt for generation.",
    )
    parser.add_argument("--metadata-file", default="", help="Optional JSONL file with prompts.")
    parser.add_argument("--prompt-index", type=int, default=0, help="Index of prompt in metadata file.")
    parser.add_argument("--seed-a", type=int, default=101, help="Seed for Candidate A.")
    parser.add_argument("--seed-b", type=int, default=202, help="Seed for Candidate B.")
    parser.add_argument(
        "--blend-modes",
        type=lambda value: [v.strip() for v in value.split(",")],
        default=["slerp", "lerp"],
        help="Comma-separated blend modes: slerp, lerp.",
    )
    parser.add_argument(
        "--merge-steps",
        type=lambda value: [int(item.strip()) for item in value.split(",")],
        default=[3, 5, 8],
        help="Comma-separated merge denoising steps k (e.g. 3,5,8).",
    )
    parser.add_argument(
        "--mask-type",
        choices=["box", "gaussian", "split", "full"],
        default="box",
        help="Type of spatial graft mask.",
    )
    parser.add_argument(
        "--mask-box",
        default="0.1,0.5,0.9,0.95",
        help="Normalized coordinates ymin,xmin,ymax,xmax in [0, 1] defining the graft region.",
    )
    parser.add_argument(
        "--mask-center",
        default="0.5,0.75",
        help="Normalized center cy,cx in [0, 1] for gaussian mask.",
    )
    parser.add_argument(
        "--mask-radius",
        default="0.3,0.3",
        help="Normalized radius ry,rx in [0, 1] for gaussian mask.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.5,
        help="Split line position in [0, 1] for split mask.",
    )
    parser.add_argument(
        "--split-axis",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Split axis for split mask.",
    )
    parser.add_argument(
        "--softness-sigma",
        type=float,
        default=2.0,
        help="Standard deviation sigma in latent pixels for smooth Gaussian boundary falloff.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Peak merge interpolation factor in [0, 1].",
    )
    parser.add_argument("--height", type=int, default=1328, help="Native image height.")
    parser.add_argument("--width", type=int, default=1328, help="Native image width.")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Denoising steps.")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale.")
    parser.add_argument("--max-sequence-length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--grid-columns", type=int, default=4, help="Grid columns for output.")
    parser.add_argument("--lora-path", default="", help="Optional LoRA weights path.")
    parser.add_argument(
        "--lora-weight",
        default="Qwen-Image-Lightning-8steps-V1.0.safetensors",
        help="LoRA weight filename.",
    )
    parser.add_argument("--output-dir", default="", help="Directory for output images and logs.")
    parser.add_argument("--low-memory", action="store_true", help="Enable VAE tiling and slicing.")
    parser.add_argument(
        "--precision", choices=["nf4", "bf16"], default="nf4", help="Model quantization precision."
    )

    args = parser.parse_args()

    if args.metadata_file:
        metadata_path = Path(args.metadata_file)
        if not metadata_path.is_file():
            parser.error(f"metadata file not found: {args.metadata_file}")
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        if not (0 <= args.prompt_index < len(rows)):
            parser.error(f"prompt_index {args.prompt_index} out of range (0..{len(rows)-1})")
        args.prompt = rows[args.prompt_index]["prompt"]

    if args.seed_a == args.seed_b:
        parser.error("seed A and seed B must differ for cross-seed complementary fusion")
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be in [0, 1]")
    if args.softness_sigma < 0.0:
        parser.error("--softness-sigma must be non-negative")
    if any(m not in {"lerp", "slerp"} for m in args.blend_modes):
        parser.error(f"--blend-modes supports only lerp, slerp, got: {args.blend_modes}")
    if any(step < 0 or step >= args.num_inference_steps - 1 for step in args.merge_steps):
        parser.error("merge steps must be strictly before the final denoising step")

    return args


if __name__ == "__main__":
    run(parse_args())
