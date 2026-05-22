# [替换此文件] preprocess_cgr_features.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
import argparse
import logging
from tqdm import tqdm

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# --- 纯Python实现的CGR图像生成 (您的代码，保持不变) ---
def generate_cgr(sequence, size=128):
    sequence = sequence.upper().replace('T', 'U')
    corners = {
        'A': np.array([0, 0]), 'U': np.array([0, size]),
        'C': np.array([size, 0]), 'G': np.array([size, size])
    }
    img_array = np.zeros((size, size), dtype=np.float32)
    pos = np.array([size // 2, size // 2], dtype=np.float32)
    for nucleotide in sequence:
        if nucleotide in corners:
            pos = (pos + corners[nucleotide]) / 2.0
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < size and 0 <= y < size:
                img_array[y, x] += 1
    if np.max(img_array) > 0:
        img_array = (img_array / np.max(img_array)) * 255.0
    return Image.fromarray(img_array.astype(np.uint8), 'L')


# --- ResNet特征提取器 ---
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用ResNet50与我们通道中的模型保持一致性（也可以用101，但50更快）
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # 【【【 核心修正！！！ 】】】
        # 原代码: self.features = nn.Sequential(*list(resnet.children())[:-6]) # -> 64 channels
        # 修正后: 我们取到 layer1 结束。resnet.children()的前5个模块是 [conv1, bn1, relu, maxpool, layer1]
        # layer1 的输出通道数正是我们需要的 256。
        self.features = nn.Sequential(*list(resnet.children())[:5])

        self.features.eval()  # 冻结模型

        # 图像预处理保持不变
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # ResNet需要3通道输入
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img, device):
        # 将PIL图像转换为张量并移动到指定设备
        img_t = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = self.features(img_t)
        return features


def main(args):
    logger.info("--- Starting CGR Feature Preprocessing ---")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Feature tensors will be saved to: {args.output_dir}")

    try:
        df = pd.read_csv(args.csv_path)
        logger.info(f"Successfully loaded data from {args.csv_path}. Found {len(df)} samples.")
    except FileNotFoundError:
        logger.error(f"Error: The file {args.csv_path} was not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # 将模型移动到设备
    feature_extractor = ResNetFeatureExtractor().to(device)

    logger.info("Processing sequences and extracting features...")
    processed_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Features"):
        gene_id = row[args.gene_id_col]
        sequence = row[args.sequence_col]
        output_path = os.path.join(args.output_dir, f"{gene_id}.pt")

        if os.path.exists(output_path) and not args.overwrite:
            continue

        cgr_image = generate_cgr(sequence, size=128)

        # 将device参数传递给forward方法
        features_tensor = feature_extractor(cgr_image, device)
        # 现在 features_tensor 的形状会是 [1, 256, 56, 56]

        torch.save(features_tensor.cpu(), output_path)
        processed_count += 1

    logger.info(f"--- Preprocessing Complete ---")
    logger.info(f"Total samples processed: {processed_count}/{len(df)}")
    logger.info(f"Features saved in '{args.output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess RNA sequences into CGR feature tensors using ResNet.")
    parser.add_argument('--csv_path', type=str, default='rna_data.csv',
                        help='Path to the main data CSV file.')
    parser.add_argument('--sequence_col', type=str, default='Sequence', help='Name of the sequence column in the CSV.')
    parser.add_argument('--gene_id_col', type=str, default='Gene_ID', help='Name of the gene ID column in the CSV.')
    parser.add_argument('--output_dir', type=str, default='data/cgr_features',
                        help='Directory to save the feature tensors.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing feature files.')

    args = parser.parse_args()
    main(args)
