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

# export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_VISIBLE_DEVICES="3"
# export CUDA_VISIBLE_DEVICES="0,1,2,5"
# export CUDA_VISIBLE_DEVICES="3,4,6,7"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Selected free port: $port"

export MASTER_PORT=$port

DATASET="mme"
# DATASET="mmvet"

# DATASET="mmmu-val" ####
# DATASET="mmmu-dev"
# DATASET="mmmu-test"

# DATASET="mmmu-val_cot"
# DATASET="mmmu-val_cot"

# pip install Levenshtein
# DATASET="mathvista-testmini"
# DATASET="mathvista-test"

# DATASET="mmvp"
# DATASET="mmbench-dev-en" ## 
# DATASET="mmbench-test-en"


keep_ratio=1.0
keep_ratio=0.75
drop_type=attn


compressed_layers_und="0-28"
calibration_samples=1
calibration_task=understanding
# sparse_mode=random
sparse_mode=prune
# sparse_mode=moe
# record_only=False

cache_dir=/mnt/bn/seed-aws-va/shwai.he/cdt-hf/results/Bagel/cos_${calibration_task}.pt

echo "CHECKPOINT: ${CHECKPOINT}"

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=8
      shift
      ;;
    *)
      shift
      ;;
  esac
done

GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "GPUS: ${GPUS}"


# exit 0

cd /mnt/bn/seed-aws-va/shwai.he/cdt-hf
model_path=/mnt/bn/seed-aws-va/shwai.he/models/ByteDance-Seed/BAGEL-7B-MoT

if  [ ${DATASET} == "mme" ]; then


  python -m eval.vlm.eval.mme.eval_ld \
            --keep_ratio $keep_ratio \
            --calibration_samples $calibration_samples \
            --compressed_layers_und $compressed_layers_und \
            --model-path $model_path \
            --sparse_mode $sparse_mode \
            --drop_type $drop_type \
            --cache_dir $cache_dir \
            --calibration_task $calibration_task \
            # > ${model_path}/${keep_ratio}_mme.log

fi

if [ ${DATASET} == "mmvet" ]; then
    python -m eval.vlm.eval.mmvet.evaluate_mmvet --datasets mmvet \
                                                --keep_ratio $keep_ratio \
                                                # --calibration_samples $calibration_samples \
                                                # --compressed_layers_und $compressed_layers_und \
                                                # --sparse_mode $sparse_mode \
    # "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-dev-en" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_20230712 \
                                                --keep_ratio $keep_ratio \
                                                --calibration_samples $calibration_samples \
                                                --compressed_layers_und $compressed_layers_und \
                                                --sparse_mode $sparse_mode \
                                                --record_only $record_only \

      output_dir=/mnt/bn/seed-aws-va/shwai.he/cdt-hf/MMBench/$compressed_layers_und/$sparse_mode/sparsity_${keep_ratio}/samples_${calibration_samples}
      python /mnt/bn/seed-aws-va/shwai.he/cdt-hf/eval/vlm/eval/mmbench/calculate.py \
                            --output_dir $output_dir > $output_dir/result.txt

      # "${ARGS[@]:1}" 
      # > log.txt
fi

if [ ${DATASET} == "mmbench-dev-cn" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_dev_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-en" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_en_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmbench-test-cn" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets mmbench_test_cn_20231003 "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-dev" ]; then

    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_dev \
                                          --keep_ratio $keep_ratio \
                                          --calibration_samples $calibration_samples \
                                          # --compressed_layers_und $compressed_layers_und \
                                          # --sparse_mode $sparse_mode
      # "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_validation \
                                          # --keep_ratio $keep_ratio \
                                          # --calibration_samples $calibration_samples \
                                          # --compressed_layers_und $compressed_layers_und \
                                          # --sparse_mode $sparse_mode
      # "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-val_cot" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu_cot --datasets MMMU_validation_cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmmu-test" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_testmini "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mathvista-test" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_test "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.pope.evaluate_pope --datasets pope "${ARGS[@]:1}"
fi

if [ ${DATASET} == "pope_cot" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.pope.evaluate_pope --datasets pope_cot --cot "${ARGS[@]:1}"
fi

if [ ${DATASET} == "vqa-gqa-testdev" ]; then
    torchrun \
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    -m eval.vlm.eval.vqa.evaluate_vqa --datasets gqa_testdev_llava "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvp" ]; then
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmvp.evaluate_mmvp --datasets MMVP "${ARGS[@]:1}"
fi
