# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for generation evaluation scripts across BAGEL, Ming, and Qwen-Image models.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist


def print_memory(prefix=""):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    print(f"[{prefix}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Max Reserved: {max_reserved:.2f} MB")
    print()


def move_generation_input_to_device(generation_input, device):
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_layer_range(s):
    """Parse a range string like '0-28' or '0-28-2' into a list of layer indices."""
    parts = s.split('-')
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1])))
    elif len(parts) == 3:
        return list(range(int(parts[0]), int(parts[1]), int(parts[2])))
    raise ValueError(f"Invalid layer range string: {s!r}. Expected 'start-end' or 'start-end-step'.")


def get_transformer_layers(model_or_pipe, model_type):
    """Return the list of transformer decoder layers for the given model type.

    model_type: "bagel" | "ming" | "qwen"
    """
    if model_type == "bagel":
        return model_or_pipe.language_model.model.layers
    elif model_type == "ming":
        return model_or_pipe.model.model.layers
    elif model_type == "qwen":
        return model_or_pipe.text_encoder.model.language_model.layers
    raise ValueError(f"Unknown model_type: {model_type!r}. Choose from 'bagel', 'ming', 'qwen'.")


@torch.no_grad()
def prune_mlp(mlp, keep_ratio: float, compressed_layers=None, score=None):
    """Prune MLP hidden dimension by activation-importance scoring.

    compressed_layers: pass None to always prune (Ming); pass a list to prune only
        layers whose mlp.layer_idx is in the list (BAGEL / Qwen).
    score: pre-computed importance tensor; computed from act_sum/act_cnt when None.
    """
    if score is None:
        act_mean = mlp.act_sum / mlp.act_cnt
        col_norm = mlp.down_proj.weight.norm(dim=0).cpu()
        score = act_mean * col_norm

    k = int(mlp.intermediate_size * keep_ratio)
    keep = torch.topk(score, k).indices.sort().values

    in_scope = compressed_layers is None or (
        hasattr(mlp, "layer_idx") and mlp.layer_idx in compressed_layers
    )
    if in_scope:
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[keep]
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[keep]
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, keep]
        mlp.intermediate_size = k
        mlp.gate_proj.out_features = k
        mlp.up_proj.out_features = k
        mlp.down_proj.in_features = k

    if hasattr(mlp, "act_sum"):
        del mlp.act_sum
    if hasattr(mlp, "act_cnt"):
        del mlp.act_cnt

    return keep.tolist()


def layer_drop_compress(layers, run_calibration_fn, keep_ratio, drop_type, skip_mode=None):
    """Calibrate and apply cosine-similarity-based layer dropping.

    Args:
        layers: list of transformer decoder layers with record/skip instrumentation.
        run_calibration_fn: callable() that runs forward passes to populate
            layer.input / layer.residual / layer.output on each layer.
        keep_ratio: fraction of layers to retain (e.g. 0.75 drops 25%).
        drop_type: granularity of the drop – "block", "attn", or "mlp".
        skip_mode: BAGEL-only modality target ("und" or "gen").
            Pass None for Ming / Qwen-Image, which use a single-path convention
            where skip_mode holds the drop_type value directly.
    """
    for layer in layers:
        layer.record = True
        if skip_mode is not None:
            layer.skip_mode = skip_mode

    run_calibration_fn()

    cos_sims = {"block": [], "attn": [], "mlp": []}
    for layer in layers:
        cos_sims["block"].append(F.cosine_similarity(layer.input, layer.output, dim=-1).mean().item())
        cos_sims["attn"].append(F.cosine_similarity(layer.input, layer.residual, dim=-1).mean().item())
        cos_sims["mlp"].append(F.cosine_similarity(layer.residual, layer.output, dim=-1).mean().item())

    scores = torch.tensor(cos_sims[drop_type])
    print(f"cos_sim scores ({drop_type}): {scores.tolist()}")

    n_drop = int((1 - keep_ratio) * len(layers))
    _, topk_index = torch.topk(scores, n_drop)

    for i, layer in enumerate(layers):
        if skip_mode is not None:
            # BAGEL: separate skip_type (what to drop) and skip_mode (which modality)
            layer.skip_mode = skip_mode
            layer.skip_type = drop_type if i in topk_index else None
        else:
            # Ming / Qwen: skip_mode encodes both what to drop and that the layer is active
            layer.skip_mode = drop_type if i in topk_index else None
        layer.record = False
        del layer.input, layer.residual, layer.output
