import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

# 读取 Excel 文件
# file_path = "your_file.xlsx"  # ← 把这里替换成你的文件路径
# file_path = "/mnt/bn/seed-aws-va/shwai.he/cdt-hf/MMBench/sparsity_0.5/results.xlsx"
# file_path = "/mnt/bn/seed-aws-va/shwai.he/cdt-hf/MMBench/sparsity_1.0/results.xlsx"

# baseline: 87.98

# file_path = "/mnt/bn/seed-aws-va/shwai.he/cdt-hf/MMBench/sparsity_1.0/samples_5/results.xlsx"
file_path = f"{args.output_dir}/results.xlsx"


df = pd.read_excel(file_path)

# 确保列名正确（区分大小写）
if "prediction" not in df.columns or "answer" not in df.columns:
    raise ValueError("Excel 文件中必须包含 'prediction' 和 'answer' 两列")

# 计算 accuracy
total = len(df)
correct = (df["prediction"] == df["answer"]).sum()
accuracy = correct / total if total > 0 else 0.0

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.4f}")
