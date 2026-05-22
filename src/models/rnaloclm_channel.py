import torch
import torch.nn as nn
import fm
import logging
import re

logger = logging.getLogger(__name__)

# RNA-FM 模型支持的最大序列长度
MAX_LEN_RNA_FM = 1022


class TextCNNBilstmWithAttention(nn.Module):

    def __init__(self, num_classes, input_dim, num_filters, filter_sizes, hidden_size, num_layers, dropout, num_heads):
        """
        初始化模型参数。
        
        Args:
            num_classes (int): 分类类别数。
            input_dim (int): 输入特征维度 (通常是 RNA-FM 的嵌入维度)。
            num_filters (int): CNN 卷积核数量。
            filter_sizes (list): CNN 卷积核大小列表 (例如 [3, 4, 5])。
            hidden_size (int): LSTM 隐藏层维度。
            num_layers (int): LSTM 层数。
            dropout (float): Dropout 比率。
            num_heads (int): 多头注意力的头数。
        """
        super().__init__()
        # 1. TextCNN 层
        # 使用不同尺寸的卷积核提取多尺度特征
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, input_dim), padding=((k - 1) // 2, 0)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
        # 2. BiLSTM 层
        # 输入维度是所有卷积核输出通道之和
        self.bilstm = nn.LSTM(input_size=num_filters * len(filter_sizes),
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        
        # 3. Attention 层
        # 输入维度是 BiLSTM 的输出维度 (hidden_size * 2)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)
        
        # 4. 全连接分类层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # 暴露特征维度供融合模块使用
        self.feature_dim = hidden_size * 2  # [# MBT-MOD] Expose feature dimension

    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (torch.Tensor): 输入嵌入，形状 [batch_size, seq_len, input_dim]。
            
        Returns:
            logits (torch.Tensor): 分类 logits。
            features (torch.Tensor): 提取的特征序列。
        """
        # 增加通道维度以适配 Conv2d: [batch, 1, seq_len, input_dim]
        x = x.unsqueeze(1)
        
        # CNN 卷积 + ReLU + Squeeze
        # 输出形状: [batch, num_filters, seq_len]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        
        # 拼接不同卷积核的输出: [batch, total_filters, seq_len]
        x = torch.cat(x, dim=1)
        
        # 调整维度以适配 LSTM: [batch, seq_len, total_filters]
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        
        # BiLSTM 处理
        x, _ = self.bilstm(x)

        # Attention 处理
        # [# MBT-MOD] This is the feature for fusion
        features, _ = self.attention(x, x, x)  # Shape: [batch, seq_len, hidden_size * 2]

        # 池化和分类
        # 对序列维度求平均，得到固定长度的向量
        pooled_features = torch.mean(features, dim=1)
        logits = self.fc(pooled_features)

        return logits, features


class RNALocLMChannel(nn.Module):
    """
    RNALoc-LM 通道：基于预训练语言模型 (RNA-FM) 的特征提取通道。
    
    流程：
    1. 使用预训练的 RNA-FM 模型提取序列的嵌入表示。
    2. 使用下游模型 (TextCNN+BiLSTM+Attention) 进一步提取特征并进行分类。
    3. 将特征投影到统一的融合维度。
    """
    def __init__(self, config, num_classes, device, fusion_cfg):
        super().__init__()
        logger.info("Initializing RNALoc-LM Channel for MBT Fusion...")
        self.config = config
        self.device = device

        # 加载预训练的 RNA-FM 模型
        logger.info(f"Loading RNA-FM model from: {config['rna_fm_model_path']}")
        self.fm_model, self.fm_alphabet = fm.pretrained.rna_fm_t12(config['rna_fm_model_path'])
        self.fm_model.to(self.device)
        self.fm_model.eval()
        # 冻结 RNA-FM 参数，不参与训练
        for param in self.fm_model.parameters():
            param.requires_grad = False
        self.batch_converter = self.fm_alphabet.get_batch_converter()

        # 初始化下游模型
        self.downstream_model = TextCNNBilstmWithAttention(
            num_classes=num_classes,
            input_dim=config['input_dim'],
            num_filters=config['num_filters'],
            filter_sizes=config['filter_sizes'],
            hidden_size=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_heads=config['num_heads']
        ).to(self.device)

        # [# MBT-MOD] 添加投影层，将特征映射到融合维度
        feature_dim = self.downstream_model.feature_dim
        fusion_dim = fusion_cfg['fusion_dim']
        self.projection = nn.Linear(feature_dim, fusion_dim).to(device)

        logger.info("RNALoc-LM Channel Initialized with projection layer.")

    def _sanitize_and_truncate_sequence(self, sequence):
        """
        清理和截断 RNA 序列。
        
        1. 移除非 ACGU 字符。
        2. 将 T 替换为 U。
        3. 截断到 RNA-FM 支持的最大长度。
        """
        cleaned_seq = re.sub(r'[^ACGU]', '', sequence.upper().replace('T', 'U'))
        if len(cleaned_seq) > MAX_LEN_RNA_FM:
            cleaned_seq = cleaned_seq[:MAX_LEN_RNA_FM]
        return 'A' if not cleaned_seq else cleaned_seq

    def forward(self, raw_sequences):
        """
        前向传播。
        
        Args:
            raw_sequences (list): 原始 RNA 序列列表。
            
        Returns:
            dict: 包含 'logits' (辅助分类结果) 和 'features' (融合用特征)。
        """
        # 预处理序列
        processed_sequences = [self._sanitize_and_truncate_sequence(seq) for seq in raw_sequences]
        
        # 转换为 RNA-FM 输入格式
        fm_data = [(f"id_{i}", seq) for i, seq in enumerate(processed_sequences)]
        _, _, batch_tokens = self.batch_converter(fm_data)
        batch_tokens = batch_tokens.to(self.device)

        # 提取 RNA-FM 嵌入 (使用第 12 层)
        with torch.no_grad():
            results = self.fm_model(batch_tokens, repr_layers=[12], need_head_weights=False)
        embeddings = results['representations'][12]

        # [# MBT-MOD] 通过下游模型获取 logits 和特征
        logits, features = self.downstream_model(embeddings)

        # 投影特征到融合维度
        projected_features = self.projection(features)
        
        # features 是未 pooled 的序列特征。
        
        
        pooled_projected = torch.mean(projected_features, dim=1) # [batch, fusion_dim]
        final_features = pooled_projected.unsqueeze(1) # [batch, 1, fusion_dim]

        return {
            "logits": logits,
            "features": final_features
        }
