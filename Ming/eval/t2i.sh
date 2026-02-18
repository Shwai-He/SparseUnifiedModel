resume_from="your_model_path"
output_dir=eval/vlm/t2i

python eval/t2i.py --resume_from ${resume_from} --output_dir ${output_dir}