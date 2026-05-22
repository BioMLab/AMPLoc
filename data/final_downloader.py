# final_downloader.py
import os
from huggingface_hub import hf_hub_download
import logging

# --- 配置 ---
# 我们要下载的模型仓库ID
repo_id = "zhihan1996/DNABERT-2-117M"
# safetensors文件所在的PR的引用
pr_revision = "refs/pr/17"
# safetensors文件在PR中的原始名称
safetensors_original_filename = "pytorch_model.safetensors"
# 我们希望它在本地保存为的标准名称
safetensors_target_filename = "model.safetensors"
# 我们需要从主分支下载的其他配置文件
config_files = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    ".gitattributes"
]
# 所有文件最终要保存到的本地文件夹名称
save_directory = "DNABERT-2-117M_final"


# --- 主程序 ---
def main():
    print(f"--- 开始下载模型 {repo_id} 的所有必需文件 ---")

    # 1. 创建本地保存目录
    os.makedirs(save_directory, exist_ok=True)
    print(f"本地目录 '{save_directory}' 已创建。")

    # 2. 从PR中下载safetensors文件
    print(f"\n正在从PR #{pr_revision.split('/')[-1]} 下载 safetensors 文件...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=safetensors_original_filename,
            revision=pr_revision,
            local_dir=save_directory,
            local_dir_use_symlinks=False,  # 确保是完整文件，而不是链接
        )
        print(f"文件 '{safetensors_original_filename}' 已成功下载到: {downloaded_path}")

        # 3. 将其重命名为标准文件名 model.safetensors (如果需要)
        final_path = os.path.join(save_directory, safetensors_target_filename)
        if downloaded_path != final_path:
            os.rename(downloaded_path, final_path)
            print(f"文件已重命名为: {final_path}")

    except Exception as e:
        print(f"!!! 下载safetensors文件时出错: {e}")
        print("请检查您的网络连接和huggingface_hub库是否已安装 (pip install huggingface_hub)。")
        return

    # 4. 从主分支下载所有其他配置文件
    print("\n正在从主分支下载配置文件...")
    for file in config_files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=save_directory,
                local_dir_use_symlinks=False,
            )
            print(f"- 已下载: {file}")
        except Exception as e:
            print(f"- 下载 {file} 失败: {e} (这可能是一个可选文件，尝试继续...)")

    print("\n--- 所有文件下载和整理完毕！ ---")
    print(f"请将 '{save_directory}' 文件夹完整上传到您服务器的 'pretrained' 目录下。")
    print(f"最终路径应为: /mnt/LncAPNet/pretrained/{repo_id}")


if __name__ == "__main__":
    # 运行前，请确保已安装huggingface_hub: pip install huggingface_hub
    main()

