# SPDX-License-Identifier: Apache-2.0

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"

DATASET="mmmu-val"  # mmmu-val | mmvp
model_path="your_model_path"

keep_ratio=0.5
drop_type=block          # block | attn | mlp
compressed_layers_und="0-28"
calibration_samples=1
sparse_mode=prune        # prune | random

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

label=${sparse_mode}/${drop_type}/${compressed_layers_und}/${keep_ratio}/${calibration_samples}
OUTPUT_DIR=$model_path/vlm/depth/$label
mkdir -p $OUTPUT_DIR

if [ "${DATASET}" == "mmmu-val" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mmmu.evaluate_mmmu_ming_ld \
        --datasets MMMU_validation \
        --model-path $model_path \
        --keep_ratio $keep_ratio \
        --calibration_samples $calibration_samples \
        --compressed_layers_und $compressed_layers_und \
        --sparse_mode $sparse_mode \
        --drop_type $drop_type

elif [ "${DATASET}" == "mmvp" ]; then
    torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${port} \
        -m eval.vlm.eval.mmvp.evaluate_mmvp_ming \
        --datasets MMVP \
        --model-path $model_path \
        --keep_ratio $keep_ratio \
        --calibration_samples $calibration_samples \
        --compressed_layers_und $compressed_layers_und \
        --sparse_mode $sparse_mode \
        --drop_type $drop_type
fi
