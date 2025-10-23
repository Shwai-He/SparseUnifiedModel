import os
import json
import pickle
from io import BytesIO
from copy import deepcopy
from typing import (
    Any, AsyncIterable, Callable, Dict, Generator,
    List, NamedTuple, Optional, Tuple, Union,
)

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from PIL import Image
import requests
from datasets import load_dataset
from safetensors.torch import load_file
from accelerate import (
    init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
)

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens

from modeling.autoencoder import load_ae
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel import (
    Bagel, BagelConfig,
    Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.bagel.qwen2_navit import NaiveCache
from inferencer import InterleaveInferencer

def build_bagel_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    max_mem_per_gpu: str = "80GiB",
    visual_gen: bool = True,
    visual_und: bool = True,
    verbose: bool = True,
):
    """
    Build and load the BAGEL multimodal model with correct configs and device mapping.

    Args:
        model_path (str): Path to the model checkpoint directory.
        dtype (torch.dtype): Data type for model weights (default: bfloat16).
        max_mem_per_gpu (str): Max memory per GPU, e.g., '80GiB'.
        visual_gen (bool): Enable visual generation module.
        visual_und (bool): Enable visual understanding module.
        verbose (bool): Print progress information.

    Returns:
        model (Bagel): Loaded BAGEL model (dispatched to GPUs)
        tokenizer (Qwen2Tokenizer): Tokenizer with added special tokens
        vae_model (torch.nn.Module): Loaded VAE model
        vae_transform, vit_transform (ImageTransform): Image processing pipelines
        new_token_ids (list[int]): Newly added token ids
    """

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config._attn_implementation = "eager"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=visual_gen,
        visual_und=visual_und,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=dtype,
        force_hooks=True,
        offload_folder="/tmp/offload",
    ).eval()

    if verbose:
        print("BAGEL model loaded successfully.")
        print(f"Device map: {device_map}")

    return model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids

def build_dataset(cal_task):
    if cal_task == "image_gen":
        dataset = []
        for line in open("prompts.txt", 'r'): 
            line = line.strip()
            dataset.append(line)
    elif cal_task == "understanding":
        TSV_PATH = "eval/vlm/data/mmbench/mmbench_dev_en_20231003.tsv"
        dataset = pd.read_csv(TSV_PATH, sep="\t", dtype=str)
    else: 
        raise NotImplementedError

    return dataset

def reset_layer_buffers(model, mode: str, target_layer: str):
    for layer in model.language_model.model.layers:
        if target_layer == "mlp":
            block = layer.mlp_moe_gen if mode == "gen" else layer.mlp
            block.sparse_mode = "prune"
            block.register_buffer("act_sum", torch.zeros(block.intermediate_size))
            block.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))
        elif target_layer == "attn":
            layer.self_attn.sparse_mode = "prune"
            act_sum_tensor = torch.zeros(layer.self_attn.num_heads)
            layer.self_attn.register_buffer("act_sum", act_sum_tensor)
            layer.self_attn.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))


def run_calibration_loop(
    inferencer,
    model,
    cal_task: str,
    dataset=None,
    target_layer="mlp",
    mode="gen",
    times=30,
    samples=30,
    save_root="data/scores_callibrated"
):

    os.makedirs(save_root, exist_ok=True)
    device = next(model.parameters()).device
    concatenated_scores = {}

    reset_layer_buffers(model, mode, target_layer)

    for j in tqdm(range(samples), desc=f"Calibrating [{cal_task}]"):

        if cal_task == "image_gen":
            prompt = dataset[j]
            inference_hyper = dict(
                cfg_text_scale=4.0,
                cfg_img_scale=1.0,
                cfg_interval=[0.4, 1.0],
                timestep_shift=3.0,
                num_timesteps=times,
                cfg_renorm_min=0.0,
                cfg_renorm_type="global",
            )
            output_dict = inferencer(text=prompt, **inference_hyper)

        elif cal_task == "understanding":
            prompt = dataset.iloc[j]["question"]
            image = dataset.iloc[j]["image"]
            inference_hyper = dict(max_think_token_n=1000, do_sample=False)
            output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
        else: 
            raise NotImplementedError

        del output_dict
        torch.cuda.empty_cache()

    all_scores = []
    for layer in model.language_model.model.layers:
        if target_layer == "mlp":
            act_sum = (layer.mlp_moe_gen if mode == "gen" else layer.mlp).act_sum
        else:
            act_sum = layer.self_attn.act_sum
        all_scores.append(act_sum.unsqueeze(0))

    all_scores = torch.cat(all_scores, dim=0).cpu().numpy()

    save_dir = f"{save_root}/{cal_task}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{target_layer}_concatenated_scores_{mode}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(all_scores, f)

    print(f"Calibration complete. Results saved to: {save_root}/{cal_task}/")

    return all_scores


mode = "und"
model_path = "your_model_path"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
model, tokenizer, vae_model, vae_transform, vit_transform, new_token_ids = build_bagel_model(
    model_path,
    dtype=torch.bfloat16,
    max_mem_per_gpu="80GiB",
)
inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

prenorm_cache = {}
layerout_cache = {}
packed_text_idx = None 
concatenated_scores = {}

samples = 30
times = 30 if mode != "und" else 1
target_layer = "mlp"                   # "attn", "mlp"
save_root = "data/scores_callibrated"

calibration_tasks = ["image_gen", "understanding"]

for cal_task in calibration_tasks:
    cal_task = "understanding"      # image_gen, understanding
    if cal_task == "image_gen":
        mode = "gen"
    else: 
        mode = "und"

    dataset = build_dataset(cal_task=cal_task)
    run_calibration_loop(
        inferencer=inferencer,
        model=model,
        cal_task=cal_task,
        dataset=dataset,
        target_layer=target_layer,
        mode=mode,
        times=times,
        samples=samples,
        save_root=save_root, # ins
    )

mode = "und"
all_scores = None

for cal_task in calibration_tasks:
    file_path = f"{save_root}/{cal_task}/combined/{target_layer}_concatenated_scores_{mode}.pkl"
    scores = read_list_from_file(file_path)
    all_scores = all_scores + scores if all_scores is not None else scores

os.makedirs(f"{save_root}/UG", exist_ok=True)
save_path = f"{save_root}/UG/{target_layer}_concatenated_scores_{mode}.pkl"
with open(save_path, "wb") as f:
    pickle.dump(all_scores, f)

print(all_scores.shape)

