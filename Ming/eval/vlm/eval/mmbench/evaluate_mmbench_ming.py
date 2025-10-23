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
import base64
import itertools
import json
import os
import random
from io import BytesIO

import pandas as pd
import torch
from eval.vlm.utils import build_transform, process_conversation
from PIL import Image
from tqdm import tqdm
import pickle

from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


ds_collections = {
    'mmbench_dev_20230712': {
        'root': 'eval/vlm/data/mmbench/mmbench_dev_20230712.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_dev_cn_20231003': {
        'root': 'eval/vlm/data/mmbench/mmbench_dev_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    },
    'mmbench_dev_en_20231003': {
        'root': 'eval/vlm/data/mmbench/mmbench_dev_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'en'
    },
    'mmbench_test_cn_20231003': {
        'root': 'eval/vlm/data/mmbench/mmbench_test_cn_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'cn'
    },
    'mmbench_test_en_20231003': {
        'root': 'eval/vlm/data/mmbench/mmbench_test_en_20231003.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'test',
        'language': 'en'
    },
    'ccbench_dev_cn': {
        'root': 'eval/vlm/data/mmbench/CCBench_legacy.tsv',
        'max_new_tokens': 100,
        'min_new_tokens': 1,
        'type': 'dev',
        'language': 'cn'
    }
}


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversation = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indexes = [_['index'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversation, answers, indexes, options


class MMBenchDataset(torch.utils.data.Dataset):

    def __init__(self, root, prompt, language):
        self.df = pd.read_csv(root, sep='\t')
        self.prompt = prompt
        self.language = language

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        #####
        # print(f"image:")
        image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')

        images = [image]

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }

        hint = self.load_from_df(idx, 'hint')
        if hint is not None:
            question = hint + '\n' + question
        for key, item in options.items():
            question += f'\n{key}. {item}'
        if self.language == 'cn':
            question = question + '\n' + self.prompt['cn']
        else:
            question = question + '\n' + self.prompt['en']

        images, conversation = process_conversation(images, question) ###

        return {
            'question': question,
            'images': images,
            'conversation': conversation,
            'answer': answer,
            'index': index,
            'option': options
        }

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        try: 
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
            self._local_indices = self._get_local_indices(size, self._world_size, self._rank)
        except: 
            self._rank = 0
            self._world_size = 1
            self._local_indices = range(0, size)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def post_process(pred, option):
    pred = pred.strip()
    option_candidate = list(option.keys())
    if len(pred) == 1:
        return pred
    if len(pred) == 0:
        pred = "C"
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
    
        dataset = MMBenchDataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            language=ds_collections[ds_name]['language'],
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for _, (questions, images, conversation, answers, indexes, options) in tqdm(enumerate(dataloader)):
            
            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": conversation[0]},
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
            pred = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]


            preds = [post_process(pred, options[0])]

            for question, pred, answer, index in zip(questions, preds, answers, indexes):
                outputs.append({
                    'question': question,
                    'answer': pred,
                    'gt_answers': answer,
                    'index': int(index)
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            results_file = 'results.xlsx'
            output_path = os.path.join(args.out_dir, results_file)
            df = pd.read_table(ds_collections[ds_name]['root'])
            cur_df = df.copy()
            if 'mmbench' in ds_name:
                cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
                cur_df.insert(6, 'prediction', None)
            else:
                cur_df = cur_df.drop(columns=['category', 'image'])
                cur_df.insert(8, 'prediction', None)
            for item in merged_outputs:
                cur_df.loc[df['index'] == item['index'], 'prediction'] = item['answer']

            cur_df.to_excel(output_path, index=False, engine='openpyxl')
            print('Results saved to {}'.format(output_path))


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

    # del layer.act_sum
    # del layer.act_cnt
    
    return keep.tolist()


def fewshot_compress(keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode="prune", record=False): 
    random.seed(args.seed)

    prompt = {
        'en': "Answer with the option's letter from the given choices directly.",
        'cn': '请直接回答选项字母。'
    }
    
    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                expert.sparse_mode = "prune"
        else: 
            mlp.sparse_mode = "prune"    

    for ds_name in args.datasets:
    
        dataset = MMBenchDataset(
            root=ds_collections[ds_name]['root'],
            prompt=prompt,
            language=ds_collections[ds_name]['language'],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []

        for num, (questions, images, conversations, answers, data_ids, options) in tqdm(enumerate(dataloader)):
        # for num, (questions, images, conversations, answers, data_ids, options) in enumerate(dataloader):
            if num > calibration_samples: 
                break 

            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": conversations[0]},
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

            for k in inputs.keys():
                if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                    inputs[k] = inputs[k].to(dtype=torch.bfloat16)

            generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                eos_token_id=processor.gen_terminator,
                generation_config=generation_config,
            )

        break

    target_layer = "mlp"
    compressed_layers = compressed_layers_und
    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if i not in compressed_layers: 
            continue
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                # if torch.distributed.get_rank() == 0:
                prune(expert, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
                expert.sparse_mode = "dense"
            
        else: 
            # if torch.distributed.get_rank() == 0:
            prune(mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
            mlp.sparse_mode = "dense" 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='mmbench_dev_20230712')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='/mnt/bn/seed-aws-va/shwai.he/cdt-hf/results/Ming/MMBench')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--keep_ratio', type=float, default=1.0)
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--compressed_layers_und', type=str, default="")
    parser.add_argument('--sparse_mode', type=str, default='prune')
    parser.add_argument('--record_only', type=str, default=None)
    parser.add_argument('--num_experts', type=int, default=48)
    parser.add_argument('--num_shared_experts', type=int, default=16)
    parser.add_argument('--top_k', type=int, default=16)

    args = parser.parse_args()

    # args.keep_ratio = (args.num_shared_experts + args.top_k) / (args.num_shared_experts + args.num_experts)

    args.out_dir = os.path.join(args.out_dir, f"{args.datasets}/{args.compressed_layers_und}/{args.sparse_mode}_{args.keep_ratio}/samples_{args.calibration_samples}")

    results_file = 'results.xlsx'
    output_path = os.path.join(args.out_dir, results_file)
    if os.path.exists(output_path):
        print(f"[test] {output_path} exists, skip")
        exit()
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    # if torch.distributed.get_rank() == 0:  
    #     print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model_path = "/mnt/bn/seed-aws-va/shwai.he/models/inclusionAI/Ming-Lite-Omni-1.5" ####
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

    image_transform = build_transform()

    prompt = {
        'en': "Answer with the option's letter from the given choices directly.",
        'cn': '请直接回答选项字母。'
    }

    if args.keep_ratio < 1.0:
        if args.sparse_mode in ["random", "prune"]:
            compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1])))
            compressed_layers_gen = list(range(28, 28))
            calibration_samples = args.calibration_samples
            sparse_mode = args.sparse_mode

            fewshot_compress(args.keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode)

        elif args.sparse_mode == "moe": 
            # ########################### Transform dense models to MoE ###########################
            def read_list_from_file(file_path):
                with open(file_path, 'rb') as file:
                    return pickle.load(file)

            gen = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/image_gen/combined/mlp_concatenated_scores_gen.pkl"
            und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/UG/combined/mlp_concatenated_scores_und.pkl"
            und = f"/mnt/bn/seed-aws-va/shwai.he/cdt-hf/data/heads/{task}/combined/mlp_10_concatenated_scores.pkl"

            scores_gen = read_list_from_file(gen)
            scores_und = read_list_from_file(und)

            start_layer, end_layer = args.compressed_layers_und.split("-")[0], args.compressed_layers_und.split("-")[1]
            start_layer, end_layer = int(start_layer), int(end_layer)
            transform_layers = list(range(start_layer, end_layer))

            transform_mode = "und"
            shared_ratio = args.num_shared_experts / (args.num_shared_experts + args.num_experts)

            args.keep_ratio
            # if resume_from is None:
            for i, layer in enumerate(model.language_model.model.layers):
                if i not in transform_layers:
                    continue
                if torch.distributed.get_rank() == 0:  
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

    torch.distributed.barrier()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9

    if torch.distributed.get_rank() == 0:  
        print(f'[test] total_params: {total_params}B')
    # exit()
    evaluate_chat_model()
