# from huggingface_hub import Repository

# repo = Repository("./model", clone_from="tianchuang/Qwen-3-8B-RHEA-property-predictor")

import os
from huggingface_hub import HfApi

# 1. 配置你的信息
# 强烈建议将本地模型文件夹直接命名为你的远程仓库名，或者指定它的绝对路径
local_folder = "./checkpoint-644"  # 你本地存放 15G 模型的文件夹路径
repo_id = "tianchuang/Qwen-3-8B-RHEA-property-predictor"  # 你的远程仓库 ID

# 如果你在国内，取消下面这行注释，开启镜像站加速（非常重要！）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 实例化 API 对象
api = HfApi()

print(f"正在上传文件夹 '{local_folder}' 到仓库 '{repo_id}'...")
print("大文件会自动切片并支持断点续传，请耐心等待...")

# 3. 一键上传整个文件夹
# 注意：运行此脚本前，请确保你已经在终端运行过 `huggingface-cli login` 并成功登录。
# 或者你也可以直接在下面传入 token 参数：api.upload_folder(..., token="你的hf_token")
try:
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",  # 说明是模型仓库
        # token=\"你的hf_token\",  # 建议通过 huggingface-cli login 登录，或使用环境变量 HF_TOKEN，不要硬编码 token
        commit_message="first model",
    )
    print("🎉 上传成功！")
except Exception as e:
    print(f"❌ 上传过程中遇到错误: {e}")
    print("💡 提示：如果是网络波动导致中断，直接重新运行此脚本即可，它会从断点处继续上传。")