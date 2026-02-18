# Copyright (c) 2023 OpenGVLab

# SPDX-License-Identifier: MIT
#
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,7"

export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0"

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Selected free port: $port"

export MASTER_PORT=$port

DATASET=${1}
# DATASET="mmvet"
# DATASET="mmmu-val"
# DATASET="mmmu-val_cot"
# DATASET="mmmu-val_cot"
# DATASET="mathvista-testmini"

DATASET="mmvp"
# DATASET="mmbench-dev-en"
DATASET="mme"
# DATASET="mmmu-val" ####

DATASETs=("mmmu-val" "mmbench-dev-en" "mmvp" "mme")
DATASETs=("mmbench-dev-en")


Keep_Ratio=(1.0 0.25 0.5 0.75)
# Keep_Ratio=(0.5 0.75)
Keep_Ratio=(0.0)

# Samples=(1 5 10 20)
Samples=(1)

compressed_layers_und=(
                        # "0-28" 
                        "14-28" 
                        "0-14" 
                        "21-28" 
                      ) 

sparse_mode="prune"
# sparse_mode="moe"
# sparse_mode="random"


echo "CHECKPOINT: ${CHECKPOINT}"

# Save original arguments
ARGS=("$@")

cd Ming

for DATASET in "${DATASETs[@]}"; do
    for keep_ratio in "${Keep_Ratio[@]}"; do
        for calibration_samples in "${Samples[@]}"; do
            for compressed_layers_und in "${compressed_layers_und[@]}"; do
                wait
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

                if  [ ${DATASET} == "mme" ]; then

                    python -m eval.vlm.eval.mme.eval_ming \
                                --keep_ratio $keep_ratio \
                                --calibration_samples $calibration_samples \
                                --compressed_layers_und $compressed_layers_und \
                                --sparse_mode $sparse_mode > results/Ming/MME/log.txt

                fi

                if [ ${DATASET} == "mmvet" ]; then
                    python -m eval.vlm.eval.mmvet.evaluate_mmvet_ming --datasets mmvet \
                                                                --keep_ratio $keep_ratio \
                                                                --calibration_samples $calibration_samples \
                                                                --compressed_layers_und $compressed_layers_und \
                                                                --sparse_mode $sparse_mode \
                                                                "${ARGS[@]:1}" 
                fi

                if [ ${DATASET} == "mmbench-dev-en" ]; then
                    datasets=mmbench_dev_20230712

                    torchrun \
                    --nnodes=$ARNOLD_WORKER_NUM \
                    --node_rank=$ARNOLD_ID \
                    --master_addr=$ARNOLD_WORKER_0_HOST \
                    --nproc_per_node=${GPUS} \
                    --master_port=${MASTER_PORT} \
                    -m eval.vlm.eval.mmbench.evaluate_mmbench_ming --datasets $datasets \
                                                                --keep_ratio $keep_ratio \
                                                                --calibration_samples $calibration_samples \
                                                                --compressed_layers_und $compressed_layers_und \
                                                                --sparse_mode $sparse_mode \
                    
                    output_dir=results/Ming/MMBench/$datasets/$compressed_layers_und/${sparse_mode}_${keep_ratio}/samples_${calibration_samples}_no_share
                    rm $output_dir/result.txt

                    python eval/vlm/eval/mmbench/calculate.py \
                                            --output_dir $output_dir > $output_dir/result.txt

                fi

                if [ ${DATASET} == "mmbench-dev-cn" ]; then
                    datasets=mmbench_dev_20230712

                    torchrun \
                    --nnodes=$ARNOLD_WORKER_NUM \
                    --node_rank=$ARNOLD_ID \
                    --master_addr=$ARNOLD_WORKER_0_HOST \
                    --nproc_per_node=${GPUS} \
                    --master_port=${MASTER_PORT} \
                    -m eval.vlm.eval.mmbench.evaluate_mmbench --datasets $datasets "${ARGS[@]:1}"
                fi

                if [ ${DATASET} == "mmbench-test-en" ]; then
                    torchrun \
                    --nnodes=$ARNOLD_WORKER_NUM \
                    --node_rank=$ARNOLD_ID \
                    --master_addr=$ARNOLD_WORKER_0_HOST \
                    --nproc_per_node=${GPUS} \
                    --master_port=${MASTER_PORT} \
                    -m eval.vlm.eval.mmbench.evaluate_mmbench_ming --datasets mmbench_test_en_20231003 "${ARGS[@]:1}"
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
                                                        --compressed_layers_und $compressed_layers_und \
                                                        --sparse_mode $sparse_mode \

                                                        # --compressed_layers_und $compressed_layers_und \
                                                        # --sparse_mode $sparse_mode
                    # "${ARGS[@]:1}"
                fi

                if [ ${DATASET} == "mmmu-val" ]; then

                    wait 
                    sparse_mode=prune
                    datasets=MMMU_validation
                    output_dir=results/Ming/MMMU/$datasets/$compressed_layers_und/${sparse_mode}_${keep_ratio}/samples_${calibration_samples}
                    echo $output_dir/log.txt

                    mkdir -p  $output_dir
                    
                    torchrun \
                    --nnodes=$ARNOLD_WORKER_NUM \
                    --node_rank=$ARNOLD_ID \
                    --master_addr=$ARNOLD_WORKER_0_HOST \
                    --nproc_per_node=${GPUS} \
                    --master_port=${MASTER_PORT} \
                    -m eval.vlm.eval.mmmu.evaluate_mmmu_ming --datasets MMMU_validation \
                                                        --keep_ratio $keep_ratio \
                                                        --calibration_samples $calibration_samples \
                                                        --compressed_layers_und $compressed_layers_und \
                                                        --sparse_mode $sparse_mode > $output_dir/log.txt

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
                    -m eval.vlm.eval.mmvp.evaluate_mmvp_ming --datasets MMVP \
                                                --keep_ratio $keep_ratio \
                                                --calibration_samples $calibration_samples \
                                                --compressed_layers_und $compressed_layers_und \
                                                --sparse_mode $sparse_mode \
                                                "${ARGS[@]:1}"
                fi

                # exit 0

            done
        done
    done
done