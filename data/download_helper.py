# download_helper.py
from transformers import AutoModel, AutoTokenizer
import os

# 要下载的模型名称
model_name = "zhihan1996/DNABERT-2-117M"
# 想要保存到的本地目录名
save_directory = "pretrained/DNABERT-2-117M"

print(f"开始下载模型和分词器: {model_name}")

# 创建目标目录
os.makedirs(save_directory, exist_ok=True)

# 下载并保存分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# 下载并保存模型
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.save_pretrained(save_directory)

print(f"模型和分词器已成功下载并保存到 '{save_directory}' 目录下。")
print("请将整个 'pretrained' 文件夹上传到您的服务器。")
