# SPDX-License-Identifier: Apache-2.0

set -x

GPUS=8

port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Selected free port: $port"

model_path="your_model_path"
metadata_file=./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl
seed=42

calibration_samples=1
keep_ratio=0.5
sparse_mode=prune # random prune
compressed_layers_gen="0-0"
compressed_layers_und="0-28"

label=${sparse_mode}/${compressed_layers_und}+${compressed_layers_gen}/${keep_ratio}/${calibration_samples}/seed${seed}
OUTPUT_DIR=$model_path/geneval/width/$label

echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=$port \
    ./gen_images_mp_ming.py \
    --output_dir $OUTPUT_DIR/images \
    --metadata_file $metadata_file \
    --batch_size 1 \
    --num_images 4 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    --keep_ratio $keep_ratio \
    --calibration_samples $calibration_samples \
    --compressed_layers_und $compressed_layers_und \
    --compressed_layers_gen $compressed_layers_gen \
    --sparse_mode $sparse_mode \
    --seed $seed \
    --strategy $strategy \

# calculate score
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=$port \
    ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
    $OUTPUT_DIR/images \
    --outfile $OUTPUT_DIR/results.jsonl \
    --model-path ./eval/gen/geneval/model \

# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $OUTPUT_DIR/results.jsonl

