# Copyright (c) 2023 OpenGVLab

# SPDX-License-Identifier: MIT
#
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import json
import os
import random

import torch
from eval.vlm.utils import load_model_and_tokenizer, build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

ds_collections = {
    'mmvet': {
        'root': 'eval/vlm/data/mm-vet/images',
        'question': 'eval/vlm/data/mm-vet/llava-mm-vet.jsonl',
        'metric': None,
        'max_new_tokens': 1000,
        'min_new_tokens': 1,
    }
}


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root, data, prompt):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = json.loads(self.data[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'text'], data['question_id'], data.get('answer', None)

        image = os.path.join(self.root, image)
        image = Image.open(image).convert('RGB')
        images = [image]
        
        question = question + ' ' + self.prompt

        images, conversation = process_conversation(images, question)

        return question_id, question, images, conversation, annotation


def evaluate_chat_model():
    random.seed(args.seed)
    prompt = ''

    for ds_name in args.datasets:
        dataset = VQADataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=prompt,
        )

        outputs = {}
        for _, (question_id, question, images, conversation, annotations) in tqdm(enumerate(dataset)):
            pred = model.chat(
                tokenizer, 
                new_token_ids,
                image_transform,
                images=images,
                prompt=conversation,
                max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
            )

            outputs[f'v1_{question_id}'] = pred

        print(f'Evaluating {ds_name} ...')
        results_file = os.path.join(args.out_dir, 'results.json')
        json.dump(outputs, open(results_file, 'w'))
        print('Results saved to {}'.format(results_file))


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

def fewshot_compress(keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode="prune"): 

    random.seed(args.seed)

    if sparse_mode == "random": 
        for i, layer in enumerate(model.language_model.model.layers):
            if i in compressed_layers_und:
                layer.mlp.sparse_mode = "random"

    elif sparse_mode == "prune":

        prompt = ''

        for i, layer in enumerate(model.language_model.model.layers):
            layer.mlp.sparse_mode = "prune"    
            layer.mlp.register_buffer("act_sum", torch.zeros(layer.mlp.intermediate_size))     # ∑|h|
            layer.mlp.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数

            # layer.mlp_moe_gen.sparse_mode = "prune"
            # layer.mlp_moe_gen.register_buffer("act_sum", torch.zeros(layer.mlp_moe_gen.intermediate_size))     # ∑|h|
            # layer.mlp_moe_gen.register_buffer("act_cnt", torch.tensor(0, dtype=torch.long))       # batch 计数

        for ds_name in args.datasets:
            dataset = VQADataset(
                root=ds_collections[ds_name]['root'],
                data=ds_collections[ds_name]['question'],
                prompt=prompt,
            )

            # outputs = {}
            for i, (question_id, question, images, conversation, annotations) in tqdm(enumerate(dataset)):
                if i > calibration_samples:
                    break

                model.chat(
                    tokenizer, 
                    new_token_ids,
                    image_transform,
                    images=images,
                    prompt=conversation,
                    max_length=ds_collections[ds_name]['max_new_tokens'], # TODO: how to use ds_collections[ds_name]['min_new_tokens']
                )

        for layer in model.language_model.model.layers:
            # keep_und = 
            prune(layer.mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers_und, )
            # keep_gen = prune(layer.mlp_moe_gen, keep_ratio=keep_ratio, compressed_layers=compressed_layers_gen, )
            # data_und.append(keep_und)
            # data_gen.append(keep_gen)

        for i, layer in enumerate(model.language_model.model.layers):
            layer.mlp.sparse_mode = "dense"    

        # return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='mmvet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results/understanding/mmvet')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--keep_ratio', type=float, default=0.5)
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--compressed_layers_und', type=str, default="")
    parser.add_argument('--sparse_mode', type=str, default='prune', choices=['prune', 'random', "moe"])

    args = parser.parse_args()
    print(args.out_dir)

    args.out_dir = os.path.join(
                        args.out_dir, 
                        f"{args.sparse_mode}/sparsity_{args.keep_ratio}/samples_{args.calibration_samples}"
                        )

    print(args.out_dir)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    image_transform = build_transform()

    if args.keep_ratio < 1.0:
        # compressed_layers_und = list(range(0, 28))
        compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1])))
        compressed_layers_gen = list(range(28, 28))
        calibration_samples = 10
        sparse_mode = args.sparse_mode
        
        fewshot_compress(args.keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()
