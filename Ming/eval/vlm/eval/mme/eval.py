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

from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm
import torch
import pickle

@torch.no_grad()
def prune(layer, keep_ratio: float = 0.5, compressed_layers: list = [], ):

    if layer.act_cnt == 0:
        raise RuntimeError(
            f"[Layer {layer.layer_idx}] prune() called before any forward pass."
        )

    if not hasattr(layer, "score"):
        act_mean = layer.act_sum / layer.act_cnt
        col_norm = layer.down_proj.weight.norm(dim=0).cpu()
        score    = act_mean * col_norm
        layer.score = score
    else: 
        score = layer.score

    k = int(layer.intermediate_size * keep_ratio)
    keep = torch.topk(score, k).indices.sort().values      # ascending order
    
    if layer.layer_idx in compressed_layers:
        # --- 裁剪权重 ---
        layer.gate_proj.weight.data = layer.gate_proj.weight.data[keep]
        layer.up_proj.weight.data   = layer.up_proj.weight.data[keep]
        layer.down_proj.weight.data = layer.down_proj.weight.data[:, keep]
        # --- 更新 meta ---
        layer.intermediate_size      = k
        layer.gate_proj.out_features = k
        layer.up_proj.out_features   = k
        layer.down_proj.in_features  = k
        # dtype  = layer.gate_proj.weight.dtype
        # device = layer.gate_proj.weight.device
        # mask   = torch.zeros(layer.intermediate_size, dtype=dtype, device=device)
        # mask[keep] = 1.0                             # 1 表示保留
        # layer.register_buffer("mask", mask)          # 自动随模型搬迁

    # # --- 清空统计，避免二次剪导致 shape 不符 ---
    # layer.act_sum = torch.zeros_like(layer.act_sum[:k])
    # layer.act_cnt.zero_()
    return keep.tolist()


def fewshot_compress(args, model, keep_ratio, Compressed_Layers_UND, compressed_layers_gen, calibration_samples, sparse_mode="prune", record=False):

    if sparse_mode == "random": 
        for i, layer in enumerate(model.language_model.model.layers):
            if i in Compressed_Layers_UND:
                layer.mlp.sparse_mode = "random"

    elif sparse_mode == "prune":
        for i, layer in enumerate(model.language_model.model.layers):
            layer.mlp.sparse_mode = "prune"    
            layer.mlp.register_buffer("act_sum", torch.zeros(layer.mlp.intermediate_size))     # ∑|h|
            layer.mlp.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数

        # layer.mlp_moe_gen.sparse_mode = "prune"
        # layer.mlp_moe_gen.register_buffer("act_sum", torch.zeros(layer.mlp_moe_gen.intermediate_size))     # ∑|h|
        # layer.mlp_moe_gen.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数

        ######## Few-shot Compression ########
        for filename in os.listdir(args.root):
            fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
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
                images, conversation = process_conversation(images, question)

                response = model.chat(
                    tokenizer, 
                    new_token_ids,
                    image_transform,
                    images=images,
                    prompt=conversation,
                    max_length=20,
                )
            # break

        keep_und = []
        for layer in model.language_model.model.layers:
            keep = prune(layer.mlp, keep_ratio=keep_ratio, compressed_layers=Compressed_Layers_UND, )
            keep_und.append(keep)

        if record: 
            with open(f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/keep_indices/MME_{calibration_samples}_und.pkl", 'wb') as file:
                pickle.dump(keep_und, file)
            return

        for i, layer in enumerate(model.language_model.model.layers):
            layer.mlp.sparse_mode = "dense"    

        # return model


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='eval/vlm/eval/mme/Your_Results')
    parser.add_argument('--out-dir', type=str, default='/mnt/bn/seed-aws-va/shwai.he/cdt-hf/results/understanding/MME')
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--Compressed_Layers_UND', type=str, default="")
    parser.add_argument('--sparse_mode', type=str, default='prune', choices=['prune', 'random', "moe"])
    parser.add_argument('--record_only', type=str, default=None)
    parser.add_argument('--num_experts', type=int, default=48)
    parser.add_argument('--num_shared_experts', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=16)


    args = parser.parse_args()

    args.keep_ratio = (args.num_shared_experts + args.top_k) / (args.num_shared_experts + args.num_experts)

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    task = "mme"

    args.out_dir = os.path.join(args.out_dir, f"{args.Compressed_Layers_UND}/{args.sparse_mode}/{task}_{args.keep_ratio}/samples_{args.calibration_samples}")
    os.makedirs(args.out_dir, exist_ok=True)

    prompt = 'Answer the question using a single word or phrase.'


    if args.record_only: 
        keep_ratio = 0.5
        Compressed_Layers_UND = list(range(int(28)))
        compressed_layers_gen = list(range(int(28)))

        calibration_samples = args.calibration_samples
        sparse_mode = "prune"
        fewshot_compress(args, model, keep_ratio, Compressed_Layers_UND, compressed_layers_gen, calibration_samples, sparse_mode, record=True)
        exit()

    if args.keep_ratio < 1.0: 
        ##### Compress the model here. 
        if args.sparse_mode in ["random", "prune"]:
            Compressed_Layers_UND = list(range(0, 28))
            Compressed_Layers_UND = list(range(int(args.Compressed_Layers_UND.split('-')[0]), int(args.Compressed_Layers_UND.split('-')[1])))
            compressed_layers_gen = list(range(28, 28))

            calibration_samples = args.calibration_samples
            sparse_mode = args.sparse_mode
            fewshot_compress(args, model, args.keep_ratio, Compressed_Layers_UND, compressed_layers_gen, calibration_samples, sparse_mode)

        elif args.sparse_mode == "moe": 
            # ########################### Transform dense models to MoE ###########################
            def read_list_from_file(file_path):
                with open(file_path, 'rb') as file:
                    return pickle.load(file)

            # gen = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/image_gen/combined/mlp_concatenated_scores_gen.pkl"
            # und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/image_gen/combined/mlp_concatenated_scores_und.pkl"
            # und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/UG/combined/mlp_concatenated_scores_und.pkl"
            # und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/understanding/combined/mlp_concatenated_scores_und.pkl"
            und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/{task}/combined/mlp_10_concatenated_scores.pkl"

            # scores_gen = read_list_from_file(gen)
            scores_und = read_list_from_file(und)

            start_layer, end_layer = args.Compressed_Layers_UND.split("-")[0], args.Compressed_Layers_UND.split("-")[1]
            start_layer, end_layer = int(start_layer), int(end_layer)
            transform_layers = list(range(start_layer, end_layer))

            transform_mode = "und"
            shared_ratio = args.num_shared_experts / (args.num_shared_experts + args.num_experts)

            args.keep_ratio
            # if resume_from is None:
            for i, layer in enumerate(model.language_model.model.layers):
                if i not in transform_layers:
                    continue

                print(f"layer {i}")

                layer.convert_dense_to_sparse_moe_dual(
                                                        mode="und", 
                                                        importance_scores = scores_und[i],    
                                                        shared_ratio = shared_ratio,          
                                                        )


                # if "gen" in training_args.transform_mode:
                #     layer.convert_dense_to_sparse_moe_dual(
                #                                             mode="gen", 
                #                                             importance_scores = scores_gen[i],    
                #                                             shared_ratio = share_ratio,          
                #                                             )


            # # ########################### End of Transformation ###########################



    # for layer in model.language_model.model.layers:
    #     print(f"{layer.layer_idx} | {layer.mlp.down_proj.weight.size()}")

    print(model.language_model)
    # exit()

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
            images, conversation = process_conversation(images, question)

            response = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=20,
            )
            response = post_processing(response)
            print(img, question, gt, response, sep='\t', file=fout)
        fin.close()
        fout.close()

    os.system(f"python -m eval.vlm.eval.mme.calculation --out-dir {args.out_dir}")
