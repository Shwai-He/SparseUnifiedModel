resume_from=/mnt/bn/seed-aws-va/shwai.he/models/ByteDance-Seed/BAGEL-7B-MoT
output_dir=/mnt/bn/seed-aws-va/shwai.he/cdt-hf/eval/vlm/t2i

python /mnt/bn/seed-aws-va/shwai.he/cdt-hf/eval/t2i.py --resume_from ${resume_from} --output_dir ${output_dir}