# Copyright (c) 2023 OpenGVLab

# SPDX-License-Identifier: MIT
#
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import argparse
import itertools
import json
import os
import random

import torch
from .data_utils import CAT_SHORT2LONG, process_single_sample
from datasets import concatenate_datasets, load_dataset
from eval.vlm.utils import build_transform, process_conversation
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


ds_collections = {
    'MMMU_validation': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'validation'
    },
    'MMMU_test': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'test'
    },
    'MMMU_dev': {
        'root': 'MMMU/MMMU',
        'max_new_tokens': 10,
        'min_new_tokens': 1,
        'split': 'dev'
    },
}


def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    images = [_['images'] for _ in batches]
    conversation = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    options = [_['option'] for _ in batches]
    return questions, images, conversation, answers, data_ids, options


class MMMUDataset(torch.utils.data.Dataset):

    def __init__(self, root, split, prompt):
        # run for each subject
        sub_dataset_list = []
        for subject in tqdm(CAT_SHORT2LONG.values()):
            sub_dataset = load_dataset(root, subject, split=split, cache_dir=os.path.join(os.getcwd(), 'eval/vlm/data/MMMU/'))
            sub_dataset_list.append(sub_dataset)

        # merge all dataset
        self.data = concatenate_datasets(sub_dataset_list)
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = process_single_sample(self.data[idx])
        data_id = data['id']
        question = data['question'].strip()
        pil_images = data['image']
        question_type = data['question_type']

        choices = eval(data['options'])
        answer = data['answer'] if 'answer' in data else None

        choice_list = []
        options = {}
        multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c.strip()))
            options[multiple_choices[i]] = c.strip()
        choice_txt = '\n'.join(choice_list)
        images = []
        for idx, pil_image in enumerate(pil_images):
            if pil_image is not None:
                if idx == 0:
                    pil_image = pil_image.resize((pil_image.width * 2, pil_image.height * 2), Image.BILINEAR)
                images.append(pil_image)

        if len(choice_txt) > 0:
            question += '\n' + choice_txt
        question += '\n' + self.prompt[question_type]
        question = question.strip()

        # NOTE: Do not add <image> since <image 1> has been added
        # question = "<image>" * len(images) + "\n" + question

        images, conversation = process_conversation(images, question)

        return {
            'question': question,
            'images': images,
            'conversation': conversation,
            'answer': answer,
            'option': options,
            'data_id': data_id
        }


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
    elif len(pred) == 0:
        pred = "C"
    elif len(pred) != 1 and pred[0] in option_candidate:
        return pred[0]
    elif len(pred) != 1 and pred[0] not in option_candidate:
        for k, v in option.items():
            if v in pred:
                return k

    return pred


def evaluate_chat_model(dataloader):
    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }
    random.seed(args.seed)

    # for ds_name in args.datasets:
    #     dataset = MMMUDataset(
    #         root=ds_collections[ds_name]['root'],
    #         split=ds_collections[ds_name]['split'],
    #         prompt=prompt,
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset=dataset,
    #         sampler=InferenceSampler(len(dataset)),
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #         collate_fn=collate_fn,
    #     )

    outputs = []
    for _, (questions, images, conversations, answers, data_ids, options) in tqdm(enumerate(dataloader)):
        
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
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if len(options[0]) == 0:
            preds = [pred]
        else:
            preds = [post_process(pred, options[0])]

        for question, pred, answer, data_id in zip(questions, preds, answers, data_ids):
            outputs.append({
                'question': question,
                'answer': pred,
                'gt_answers': answer,
                'data_id': data_id
            })

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        print(f'Evaluating {ds_name} ...')
        output_path = os.path.join(args.out_dir, "prediction.json")
        outputs = {}
        for item in merged_outputs:
            outputs[item['data_id']] = item['answer']
        with open(output_path, 'w') as f:
            json.dump(outputs, f, indent=4)
        print('Results saved to {}'.format(output_path))
        if ds_collections[ds_name]['split'] == 'validation':
            print('Evaluating ...')
            cmd = f'python -m eval.vlm.eval.mmmu.main_eval_only ' \
                    f'--output_path {output_path} ' \
                    f'--answer_path eval/vlm/eval/mmmu/answer_dict_val.json ' \
                    f'--out-dir {args.out_dir}'
            print(cmd)
            os.system(cmd)
        output_path = os.path.join(args.out_dir, "results.jsonl")
        writer = open(output_path, 'w')
        for item in merged_outputs:
            writer.write(json.dumps(item) + '\n')
        writer.close()
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

def fewshot_compress(keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode="prune", dataloader=None): 
    random.seed(args.seed)
    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }
    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                expert.sparse_mode = "prune"
        else: 
            mlp.sparse_mode = "prune"    


    # for ds_name in args.datasets:
    #     dataset = MMMUDataset(
    #         root=ds_collections[ds_name]['root'],
    #         split=ds_collections[ds_name]['split'],
    #         prompt=prompt,
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset=dataset,
    #         sampler=InferenceSampler(len(dataset)),
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=True,
    #         drop_last=False,
    #         collate_fn=collate_fn,
    #     )


    for num, (questions, images, conversations, answers, data_ids, options) in tqdm(enumerate(dataloader)):
    # for num, (questions, images, conversation, answers, data_ids, options) in enumerate(dataloader):
        if num > calibration_samples: 
            break 

        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text":  conversations[0]},
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

        # break

    # exit()
    target_layer = "mlp"
    compressed_layers = compressed_layers_und
    for i, layer in enumerate(model.model.model.layers):
        mlp = layer.mlp 
        if i not in compressed_layers: 
            continue
        if hasattr(mlp, "experts"):
            for expert in mlp.experts:
                # print(expert.sparse_mode, expert.act_sum)
                # if torch.distributed.get_rank() == 0:
                prune(expert, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
                expert.sparse_mode = "dense"
            
        else: 
            # print(mlp.sparse_mode, expert.act_sum)
            # if torch.distributed.get_rank() == 0:
            prune(mlp, keep_ratio=keep_ratio, compressed_layers=compressed_layers,)
            mlp.sparse_mode = "dense" 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MMMU_validation')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results/Ming/MMMU')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument('--calibration_samples', type=int, default=1)
    parser.add_argument('--compressed_layers_und', type=str, default="0-0")
    parser.add_argument('--sparse_mode', type=str, default='prune')
    parser.add_argument('--keep_ratio', type=float, default=1.0)

    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, f"{args.datasets}/{args.compressed_layers_und}/sparsity_{args.keep_ratio}/samples_{args.calibration_samples}")
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')

    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if torch.distributed.get_rank() == 0:
        print(f"args.out_dir: {args.out_dir}")
    
    model_path = "your_model_path" ####
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

    ds_name = args.datasets[0]

    prompt = {
        'multiple-choice': "Answer with the option's letter from the given choices directly.",
        'open': 'Answer the question using a single word or phrase.'
    }

    dataset = MMMUDataset(
        root=ds_collections[ds_name]['root'],
        split=ds_collections[ds_name]['split'],
        prompt=prompt,
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


    if args.keep_ratio < 1.0:
        compressed_layers_und = list(range(int(args.compressed_layers_und.split('-')[0]), int(args.compressed_layers_und.split('-')[1])))
        compressed_layers_gen = list(range(28, 28))
        calibration_samples = args.calibration_samples
        sparse_mode = args.sparse_mode
        fewshot_compress(args.keep_ratio, compressed_layers_und, compressed_layers_gen, calibration_samples, sparse_mode, dataloader)


    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    # if torch.distributed.get_rank() == 0:  
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model(dataloader)
