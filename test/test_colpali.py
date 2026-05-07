import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from pdf2image import convert_from_path
import os

# 使用本地下载的路径 - 使用脚本所在目录的父目录作为基准
script_dir = os.path.dirname(os.path.abspath(__file__))
local_model_path = os.path.join(os.path.dirname(script_dir), "models/colqwen2.5-v0.2")

print(f"正在从本地加载模型: {local_model_path}")

# Mac Apple Silicon 配置
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float32

model = ColQwen2_5.from_pretrained(
    local_model_path,
    torch_dtype=dtype,
    device_map=device,
    local_files_only=True,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(local_model_path, local_files_only=True)

print("✅ 模型加载成功！")
print("✅ 模型加载成功！")
print(f"使用设备: {device}")
print(f"模型路径: {local_model_path}")