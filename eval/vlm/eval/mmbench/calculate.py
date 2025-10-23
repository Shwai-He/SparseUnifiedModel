import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

file_path = f"{args.output_dir}/results.xlsx"
df = pd.read_excel(file_path)

if "prediction" not in df.columns or "answer" not in df.columns:
    raise ValueError("Excel 文件中必须包含 'prediction' 和 'answer' 两列")

total = len(df)
correct = (df["prediction"] == df["answer"]).sum()
accuracy = correct / total if total > 0 else 0.0

print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.4f}")
