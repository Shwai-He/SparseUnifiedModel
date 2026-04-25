# SPDX-License-Identifier: Apache-2.0

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"

DATASET="mme"  # mme | mmbench-dev-en | mmmu-val | mmvp | mathvista-testmini
model_path="your_model_path"

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

if [ "${DATASET}" == "mme" ]; then
    python -m eval.vlm.eval.mme.eval \
        --model-path $model_path \
        --out-dir $model_path/MME

elif [ "${DATASET}" == "mmbench-dev-en" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mmbench.evaluate_mmbench \
        --datasets mmbench_dev_20230712 \
        --model-path $model_path

elif [ "${DATASET}" == "mmmu-val" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mmmu.evaluate_mmmu \
        --datasets MMMU_validation \
        --model-path $model_path

elif [ "${DATASET}" == "mmvp" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mmvp.evaluate_mmvp \
        --datasets MMVP \
        --model-path $model_path

elif [ "${DATASET}" == "mathvista-testmini" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mathvista.evaluate_mathvista \
        --datasets MathVista_testmini \
        --model-path $model_path
fi
