# src/models/rpi_channel.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    位置编码层 (Positional Encoding)。
    
    功能：
    为序列数据添加位置信息，使模型能够感知序列中元素的顺序。
    使用正弦和余弦函数生成固定的位置编码。
    """
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_encoding', pos_encoding)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_encoding[:seq_len, :]


class GuidedAttentionLayer(nn.Module):
    """
    引导式注意力层 (Guided Attention Layer)。
    
    实现核心的自注意力机制，允许模型关注序列中重要的部分（例如关键的 RBP 结合位点）。
    包含多头注意力、前馈网络、层归一化和残差连接。
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        # 多头注意力机制
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, return_attention=False):
        # 多头注意力 + 残差连接
        attn_output, attn_weights = self.multi_head_attention(x, x, x, average_attn_weights=False)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        if return_attention:
            # 平均所有头的注意力权重: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
            attn_weights = attn_weights.mean(dim=1)
            return out2, attn_weights
        return out2


class FeatureInteractionLayer(nn.Module):
    """
    特征交互层 (Feature Interaction Layer)。
    
    功能：
    通过全连接层增强特征之间的非线性交互，进一步提取高层语义特征。
    """
    
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        # 特征交互网络
        self.interaction_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差连接的归一化
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x):
        # 计算特征交互
        interaction_output = self.interaction_net(x)
        # 残差连接
        output = self.layernorm(x + interaction_output)
        return output


class RPIChannel(nn.Module):
    """
    RNA-蛋白质相互作用通道 (Multi-Source Version)。
    
    利用 RNA 与 RNA结合蛋白 (RBP) 的相互作用分数作为特征，预测 LncRNA 的亚细胞定位。
    支持多个 RPI 数据源 (e.g., RPI-Net, GGCN, LPI, SVM)，每个源作为独立的特征通道。
    """

    def __init__(self, config, num_classes, fusion_cfg):
        """
        初始化 RPIChannel。
        
        Args:
            config (dict): RPI 通道配置。
            num_classes (int): 分类类别数。
            fusion_cfg (dict): 融合配置。
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.fusion_dim = fusion_cfg['fusion_dim']
        
        # 从配置中获取RPI模型参数
        rpi_dim = config.get('rpi_dim', 22)  # RPI特征维度
        d_model = config.get('d_model', 132)  # 嵌入维度
        seq_len = config.get('seq_len', 22)  # 序列长度
        feature_dim = config.get('feature_dim', 6)  # 特征维度
        num_heads = config.get('num_heads', 2)  # 注意力头数
        num_layers = config.get('num_layers', 3)  # 注意力层数
        dropout_rate = config.get('dropout_rate', 0.1)  # Dropout率
        
        # 加载RPI和RBP数据 (Multi-Source)
        self.rpi_data_dict = {}
        self.rbp_data = None
        self._load_rpi_data()
        
        # 特征嵌入层 (Decoupled: One embedding layer per source)
        # 使用 ModuleDict 来存储每个源的独立嵌入层
        self.feature_embeddings = nn.ModuleDict()
        
        # 如果没有加载到任何源（例如初始化时），至少创建一个默认的
        source_names = list(self.rpi_data_dict.keys()) if self.rpi_data_dict else ['default']
        
        for name in source_names:
            self.feature_embeddings[name] = nn.Sequential(
                nn.Linear(rpi_dim, d_model),
                nn.ReLU(),
                nn.BatchNorm1d(d_model),
                nn.Dropout(0.2)
            )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(seq_len, feature_dim)
        
        # 多层引导式注意力层 (Shared weights)
        self.attention_layers = nn.ModuleList([
            GuidedAttentionLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # 特征交互层 (Shared weights)
        self.feature_interaction = FeatureInteractionLayer(
            d_model=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影层，将特征投影到融合维度 (Shared weights)
        self.project_to_fusion = nn.Linear(feature_dim, self.fusion_dim)
        
        # 辅助分类器 (Shared weights)
        self.aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, num_classes)
        )
        
        # Normalization layer for each source output
        self.output_norm = nn.LayerNorm(self.fusion_dim)

        # 大幅降低RPI通道的类别权重，减少其对整体性能的主导作用
        # 使用更平衡的权重分配
        class_weights = torch.tensor([
            1.0,  # Nucleus权重 (降低)
            1.0,  # Chromatin权重 (降低)
            2.0,  # Cytoplasm权重 (适度)
            3.0   # Insoluble cytoplasm权重 (适度)
        ], dtype=torch.float32)
        self.register_buffer('class_weights', class_weights)
        
        # 创建带权重的损失函数
        self.aux_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        logger.info(f"RPIChannel initialized with rpi_dim={rpi_dim}, d_model={d_model}, "
                   f"feature_dim={feature_dim}, num_heads={num_heads}, num_layers={num_layers}.")
        logger.info(f"RPI Channel class weights: {self.class_weights.tolist()}")
    
    def _load_rpi_data(self):
        """加载RPI和RBP数据 (Multi-Source)"""
        try:
            # Load RBP locations (Shared)
            rbp_path = self.config.get('rbp_file', 'data/rpi_data/rbp_locations.csv')
            self.rbp_data = pd.read_csv(rbp_path)
            logger.info(f"RBP data loaded: {self.rbp_data.shape}")

            # Load RPI Sources
            rpi_sources = self.config.get('rpi_sources', {})
            if not rpi_sources:
                # Fallback to single file for backward compatibility
                rpi_path = self.config.get('rpi_file', 'data/processed/aligned_rpi_scores.csv')
                rpi_sources = {'rpi_default': rpi_path}
                logger.warning("No 'rpi_sources' found in config, falling back to 'rpi_file'.")

            for source_name, file_path in rpi_sources.items():
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    self.rpi_data_dict[source_name] = df
                    logger.info(f"RPI Source '{source_name}' loaded: {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load RPI source '{source_name}' from {file_path}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load RPI/RBP data: {e}")
            logger.warning("RPI Channel will use dummy data")
    
    def _get_rpi_features(self, gene_ids: List[str], source_name: str) -> torch.Tensor:
        """根据基因ID获取RPI特征 (Specific Source)"""
        rpi_data = self.rpi_data_dict.get(source_name)
        
        if rpi_data is None:
            # 如果没有RPI数据，返回随机特征
            batch_size = len(gene_ids)
            rpi_dim = self.config.get('rpi_dim', 22)
            return torch.randn(batch_size, rpi_dim)
        
        features = []
        missing_count = 0
        for gene_id in gene_ids:
            if gene_id in rpi_data.index:
                data = rpi_data.loc[gene_id]
                if isinstance(data, pd.DataFrame):
                    # Handle duplicates: take the mean
                    feature = data.values.mean(axis=0)
                else:
                    feature = data.values
                features.append(feature)
            else:
                # 如果基因ID不存在，使用零向量
                rpi_dim = len(rpi_data.columns)
                features.append(np.zeros(rpi_dim))
                missing_count += 1
        
        # 记录缺失的基因ID数量 (Only log for first batch to avoid spam, or use debug)
        # if missing_count > 0:
        #     logger.debug(f"[{source_name}] Found {missing_count}/{len(gene_ids)} gene IDs not in RPI data.")
        
        # 使用numpy.array()先转换为numpy数组，确保数据类型为float32
        features_array = np.array(features, dtype=np.float32)
        return torch.tensor(features_array, dtype=torch.float32)
    
    def forward(self, gene_ids: List[str], return_attention=False):
        """
        前向传播 (Multi-Source).
        
        Args:
            gene_ids (List[str]): 基因ID列表。
            return_attention (bool): 是否返回注意力权重用于分析。
            
        Returns:
            dict: 包含 'features' (Dict of tensors) 和 'logits' (Dict of logits).
        """
        features_dict = {}
        logits_dict = {}
        
        # Iterate over all loaded RPI sources
        for source_name in self.rpi_data_dict.keys():
            # 1. Get Raw Features
            rpi_features = self._get_rpi_features(gene_ids, source_name)
            
            # 确保特征在正确的设备上
            device = next(self.parameters()).device
            rpi_features = rpi_features.to(device)
            
            # 2. Feature Embedding (Decoupled)
            # Use the specific embedding layer for this source
            if source_name in self.feature_embeddings:
                embedded = self.feature_embeddings[source_name](rpi_features)
            else:
                # Fallback (should not happen if init is correct)
                # Use the first available embedding or raise error
                first_key = list(self.feature_embeddings.keys())[0]
                embedded = self.feature_embeddings[first_key](rpi_features)
            
            # 重塑为序列形式
            seq_len = self.config.get('seq_len', 22)
            feature_dim = self.config.get('feature_dim', 6)
            embedded = embedded.view(-1, seq_len, feature_dim)  # [batch_size, seq_len, feature_dim]
            
            # 位置编码
            embedded = self.positional_encoding(embedded)
            
            # 3. Guided Attention (Shared)
            for attention_layer in self.attention_layers:
                embedded = attention_layer(embedded)
            
            # 4. Feature Interaction (Shared)
            embedded = self.feature_interaction(embedded)
            
            # 5. Global Pooling & Aux Classification
            pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)  # [batch_size, feature_dim]
            logits = self.aux_classifier(pooled)
            logits_dict[source_name] = logits
            
            # 6. Project to Fusion Dim
            features_for_fusion = self.project_to_fusion(embedded)  # [batch_size, seq_len, fusion_dim]
            
            # 7. Normalization (Crucial for stability across sources)
            features_for_fusion = self.output_norm(features_for_fusion)
            
            features_dict[source_name] = features_for_fusion
        
        result = {
            "features": features_dict, # Dict of [Batch, Seq, FusionDim]
            "logits": logits_dict      # Dict of [Batch, NumClasses]
        }
            
        return result
