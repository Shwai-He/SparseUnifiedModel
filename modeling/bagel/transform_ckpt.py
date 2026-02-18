import torch
from safetensors.torch import load_file, save_file  # 如果你使用 safetensors 格式

# === Step 1: 读取原始 FP32 的 state_dict ===
state_dict_path = "hf/BAGEL-7B-MoT/ema.safetensors"  # 或者 model_fp32.safetensors
# state_dict_path = "hf/BAGEL-7B-MoT/ema_bf16.safetensors"  # 或者 model_fp32.safetensors

state_dict_path = "hf/BAGEL-7B-MoT/ae.safetensors"

use_safetensors = state_dict_path.endswith(".safetensors")

if use_safetensors:
    state_dict = load_file(state_dict_path, device="cpu")
else:
    state_dict = torch.load(state_dict_path, map_location="cpu")

for key in state_dict: 
    print(state_dict[key].dtype)

exit()

# === Step 2: 转换所有 float32 参数为 bfloat16 ===
bf16_state_dict = {
    k: v.to(torch.bfloat16) if torch.is_floating_point(v) and v.dtype == torch.float32 else v
    for k, v in state_dict.items()
}


# === Step 3: 保存新的 BF16 checkpoint ===
out_path = state_dict_path.replace(".safetensors","_bf16.safetensors") if use_safetensors else state_dict_path.replace(".pth", "_bf16.pth")

if use_safetensors:
    save_file(bf16_state_dict, out_path)
else:
    torch.save(bf16_state_dict, out_path)

print(f"Saved BF16 checkpoint to: {out_path}")
