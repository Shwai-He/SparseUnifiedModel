# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import os
import re

from eval.vlm.utils import build_transform, process_conversation
from PIL import Image
from tqdm import tqdm
import torch
import pickle
import random
from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


@torch.no_grad()
def prune(layer, keep_ratio: float = 0.5, compressed_layers: list = [], ):

    # print(layer.sparse_mode, layer.act_sum)
    # if layer.act_cnt == 0:
    #     raise RuntimeError(
    #         f"[Layer {layer.layer_idx}] prune() called before any forward pass."
    #     )

    if not hasattr(layer, "score"):
        act_mean = layer.act_sum / layer.act_cnt
        col_norm = layer.down_proj.weight.norm(dim=0).cpu()
        score    = act_mean * col_norm
        layer.score = score
    else: 
        score = layer.score

    k = int(layer.intermediate_size * keep_ratio)
    keep = torch.topk(score, k).indices.sort().values      # ascending order
    
    # if layer.layer_idx in compressed_layers:
    # --- 裁剪权重 ---
    layer.gate_proj.weight.data = layer.gate_proj.weight.data[keep]
    layer.up_proj.weight.data   = layer.up_proj.weight.data[keep]
    layer.down_proj.weight.data = layer.down_proj.weight.data[:, keep]
    # --- 更新 meta ---
    layer.intermediate_size      = k
    layer.gate_proj.out_features = k
    layer.up_proj.out_features   = k
    layer.down_proj.in_features  = k

    del layer.act_sum
    del layer.act_cnt
    
    return keep.tolist()

def fewshot_compress(keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode="prune", record=False): 
    random.seed(args.seed)
    prompt = "Answer with the option's letter from the given choices directly."

    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                expert.sparse_mode = "prune"
        else: 
            mlp.sparse_mode = "prune"    

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for i, line in enumerate(tqdm(lines)):
            if i > calibration_samples: 
                break
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, img)
            if not os.path.exists(img_path):
                img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, "images", img)
            if not os.path.exists(img_path):
                continue
            images = [Image.open(img_path).convert('RGB')]
            # images, conversation = process_conversation(images, question)
            conversation = question

            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": prompt + conversation},
                    ],
                },
            ]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                return_tensors="pt",
            )

            inputs = inputs.to(model.device)
            generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=processor.gen_terminator,
                generation_config=generation_config,
            )

        break

    # exit()
    target_layer = "mlp"
    compressed_layers = compressed_layers_und
    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if i not in compressed_layers: 
            continue
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                prune(expert, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
                expert.sparse_mode = "dense"
            
        else: 
            prune(mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
            mlp.sparse_mode = "dense" 


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='eval/vlm/eval/mme/Your_Results')
    parser.add_argument('--out-dir', type=str, default='/mnt/bn/seed-aws-va/shwai.he/cdt-hf/results/Ming/MME')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--compressed_layers_und', type=str, default="")
    parser.add_argument('--sparse_mode', type=str, default='prune', choices=['prune', 'random', "moe"])
    parser.add_argument('--record_only', type=str, default=None)
    parser.add_argument('--num_experts', type=int, default=48)
    parser.add_argument('--num_shared_experts', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=16)


    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, f"{args.compressed_layers_und}/{args.sparse_mode}_{args.keep_ratio}/samples_{args.calibration_samples}")

    # args.keep_ratio = (args.num_shared_experts + args.top_k) / (args.num_shared_experts + args.num_experts)

    model_path = "/mnt/bn/seed-aws-va/shwai.he/models/inclusionAI/Ming-Lite-Omni-1.5" ####
    image_transform = build_transform()

    task = "mme"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
        load_image_gen=True,
        low_cpu_mem_usage=True       # Minimize CPU memory during loading
    )

    device = "cuda"
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)

    print(f"args.out_dir: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    prompt = 'Answer the question using a single word or phrase.'

    if args.keep_ratio < 1.0: 
        ##### Compress the model here. 
        if args.sparse_mode in ["random", "prune"]:
            compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1])))
            compressed_layers_gen = list(range(28, 28))

            calibration_samples = args.calibration_samples
            sparse_mode = args.sparse_mode
            fewshot_compress(args.keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode)

        else: 
            raise NotImplementedError

    total_params = sum(p.numel() for p in model.parameters()) / 1e9

    print(f'[test] total_params: {total_params}B')
    ######## 

    ######## Testing ########
    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(args.out_dir, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, question, gt = line.strip().split('\t')
            question = question + ' ' + prompt
            img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, img)
            if not os.path.exists(img_path):
                img_path = os.path.join('eval/vlm/data/mme/MME_Benchmark_release_version', filename, "images", img)
            if not os.path.exists(img_path):
                continue
            images = [Image.open(img_path).convert('RGB')]
            # images, conversation = process_conversation(images, question)
            conversation = question

            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": prompt + conversation},
                    ],
                },
            ]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                audios=audio_inputs,
                return_tensors="pt",
            )

            inputs = inputs.to(model.device)
            generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=processor.gen_terminator,
                generation_config=generation_config,
            )
            generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            response = post_processing(response)
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()

    os.system(f"python -m eval.vlm.eval.mme.calculation --out-dir {args.out_dir}")
