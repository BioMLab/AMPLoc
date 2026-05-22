import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CFPLncLocChannel(nn.Module):
    """
    CFPLncLocChannel 模型：用于处理 CGR (Chaos Game Representation) 图像特征的通道。
    该模型使用卷积神经网络 (CNN) 提取图像特征，并将其投影到融合空间，
    同时提供一个辅助分类头用于单独的监督信号。
    """
    def __init__(self, config, num_classes, fusion_cfg):
        super().__init__()
        # 从配置中读取参数
        in_channels = config['in_channels']  # 输入通道数，通常为 256 (例如来自预训练模型的特征图)
        num_filters = config['num_filters']  # 卷积层的滤波器数量 (输出通道数)，例如 32
        dense_dim = config['dense_dim']  # 全连接层的维度，例如 1024
        fusion_dim = fusion_cfg['fusion_dim'] # 融合空间的维度，用于与其他模态对齐

        # 定义卷积块：包含两个卷积层，每个后面跟着 ReLU 激活和最大池化
        # 旨在从输入的 CGR 图像特征中提取空间模式
        self.conv_block = nn.Sequential(
            # 第一层卷积：改变通道数，保持空间尺寸 (padding=1)
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1), 
            nn.ReLU(), # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2), # 下采样，尺寸减半
            
            # 第二层卷积：进一步提取特征
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 再次下采样，尺寸减半
        )
        
        # 展平层：将多维特征图展平为一维向量，以便输入全连接层
        self.flatten = nn.Flatten()

        # [# MBT-MOD] 全连接层 1：将展平后的特征映射到高维特征空间 (dense_dim)
        # 输入维度计算：num_filters * 14 * 14 是假设经过两次池化后的特征图大小
        # 注意：这里硬编码了 14*14，意味着输入图像大小应该是固定的 (例如 56x56 -> 28x28 -> 14x14)
        self.fc1 = nn.Linear(num_filters * 14 * 14, dense_dim)

        # Dropout 层：防止过拟合
        self.dropout = nn.Dropout(config['dropout'])

        # [# MBT-MOD] 辅助分类头：直接从该通道的特征进行分类
        # 这有助于在训练早期提供梯度，并确保该模态学习到有判别力的特征
        self.aux_head = nn.Linear(dense_dim, num_classes)

        # [# MBT-MOD] 投影层：将特征投影到统一的融合维度
        # 这是为了多模态融合 (如 MBT Fusion) 做准备
        self.projection = nn.Linear(dense_dim, fusion_dim)

        logger.info("CFPLncLocChannel initialized for MBT Fusion.")

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入张量，形状通常为 [batch_size, in_channels, height, width]
               或者是 [batch_size, 1, in_channels, height, width] (需要 squeeze)
        Returns:
            dict: 包含 'logits' (辅助分类结果) 和 'features' (用于融合的投影特征)
        """
        # 处理可能的额外维度：如果输入是 5D 张量且第 1 维为 1，则压缩该维度
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)

        # 通过卷积块提取特征
        x = self.conv_block(x)
        
        # 展平特征图
        x = self.flatten(x)

        # [# MBT-MOD] 通过全连接层生成高维特征，并使用 ReLU 激活
        features = torch.relu(self.fc1(x))  # Shape: [batch, dense_dim]

        # 计算辅助 logits (用于辅助损失)
        # 先经过 dropout，再通过辅助分类头
        x_for_logits = self.dropout(features)
        logits = self.aux_head(x_for_logits)

        # 将特征投影到融合维度
        projected_features = self.projection(features)

        # [# MBT-MOD] 调整特征形状以适配融合模块
        # 增加一个序列维度 (seq_len=1)，使其形状变为 [batch, 1, fusion_dim]
        # 这样可以被视为序列长度为 1 的特征序列，方便与 Transformer 等模型结合
        projected_features = projected_features.unsqueeze(1)  # Shape: [batch, 1, fusion_dim]

        return {
            "logits": logits,
            "features": projected_features
        }
