
import torch
import torch.nn as nn
import logging
import random

logger = logging.getLogger(__name__)

class RobustMBT_Fuser(nn.Module):
    """
    RobustMBT_Fuser: 多模态瓶颈变换器 (Robust Multimodal Bottleneck Transformer)。
    """
    def __init__(self, config):
        """
        初始化 RobustMBT_Fuser。
        
        Args:
            config (dict): 包含融合器参数的配置字典。
                - fusion_dim: 融合特征维度。
                - num_layers: Transformer 层数。
                - num_heads: 注意力头数。
                - num_fusion_tokens: 瓶颈 Token (Bottleneck Tokens) 的数量。
                - modality_dropout_prob: 模态 Dropout 的概率 (0.0 - 1.0)。
        """
        super().__init__()
        fusion_dim = config['fusion_dim']
        num_layers = config['num_layers']
        num_heads = config['num_heads']
        num_fusion_tokens = config['num_fusion_tokens']
        ffn_dim = config.get('ffn_dim', fusion_dim * 4)
        dropout = config.get('dropout', 0.1)
        
        # Modality Dropout Probability (模态丢弃概率)
        self.modality_dropout_prob = config.get('modality_dropout_prob', 0.0)
        self.num_layers = num_layers

        # Learnable fusion tokens (可学习的融合 Token)
        # 这些 Token 将作为不同模态之间信息交换的 "瓶颈"
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, fusion_dim))

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(num_layers)])
        self.self_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, fusion_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        logger.info(f"RobustMBT_Fuser initialized. Modality Dropout Prob: {self.modality_dropout_prob}")

    def _coerce_batch_vector(self, weight, batch_size, device):
        if isinstance(weight, torch.Tensor):
            weight = weight.to(device)
            if weight.dim() == 0:
                return weight.expand(batch_size)
            if weight.dim() == 1:
                return weight
            if weight.dim() == 2:
                return weight.squeeze(-1) if weight.size(-1) == 1 else weight.mean(dim=-1)
            return weight.view(weight.size(0), -1).mean(dim=1)

        return torch.full((batch_size,), float(weight), device=device)

    def _build_weight_matrix(self, features_list, channel_names, channel_weights, policy_bundle):
        batch_size = features_list[0].shape[0]
        device = features_list[0].device
        num_modalities = len(features_list)

        static_weights = []
        for idx in range(num_modalities):
            channel_name = channel_names[idx] if channel_names is not None else str(idx)
            base_weight = 1.0
            if channel_weights is not None and channel_name in channel_weights:
                base_weight = channel_weights[channel_name]
            static_weights.append(self._coerce_batch_vector(base_weight, batch_size, device).clamp(min=0.0))

        weight_matrix = torch.stack(static_weights, dim=1)

        if policy_bundle is not None:
            policy_scores = policy_bundle.get('action_mean')
            effective_mask = policy_bundle.get('effective_mask')

            if isinstance(policy_scores, torch.Tensor):
                policy_scores = policy_scores.to(device)
                if policy_scores.dim() == 3:
                    policy_scores = policy_scores.view(policy_scores.size(0), -1)
                if policy_scores.dim() == 2 and policy_scores.size(1) == num_modalities:
                    weight_matrix = weight_matrix * policy_scores.clamp(min=0.0)

            if isinstance(effective_mask, torch.Tensor):
                effective_mask = effective_mask.to(device)
                if effective_mask.dim() == 3:
                    effective_mask = effective_mask.view(effective_mask.size(0), -1)
                if effective_mask.dim() == 2 and effective_mask.size(1) == num_modalities:
                    weight_matrix = weight_matrix * (0.5 + 0.5 * effective_mask.clamp(min=0.0))

        weight_matrix = torch.clamp(weight_matrix, min=0.0)

        if policy_bundle is not None:
            temperature = policy_bundle.get('hparams', {}).get('temperature', 1.0)
            temperature = self._coerce_batch_vector(temperature, batch_size, device).clamp(min=1e-3)
        else:
            temperature = torch.ones(batch_size, device=device)

        normalized = torch.softmax(torch.log(weight_matrix + 1e-6) / temperature.unsqueeze(1), dim=1)
        return normalized

    def forward(self, features_list, channel_names=None, channel_weights=None, policy_bundle=None):
        """
        前向传播函数。
        
        Args:
            features_list: 包含各模态特征张量的列表，每个张量形状为 [B, S_i, D]。
            channel_names: 对应特征列表的通道名称列表。
            channel_weights: 各通道的权重字典。
            
        Returns:
            fused_bottleneck: 融合后的瓶颈特征，形状为 [B, num_fusion_tokens, D]。
        """
        batch_size = features_list[0].shape[0]
        device = features_list[0].device
        num_modalities = len(features_list)

        modality_gate = self._build_weight_matrix(features_list, channel_names, channel_weights, policy_bundle)

        # Apply channel weights and Modality Dropout
        # 应用通道权重和模态 Dropout
        processed_features = []
        keep_mask = [1.0] * num_modalities
        
        if self.training and self.modality_dropout_prob > 0:
            # Generate mask for each modality
            # 生成模态掩码：1 = 保留, 0 = 丢弃
            # Randomly drop modalities (随机丢弃模态)
            for i in range(num_modalities):
                if random.random() < self.modality_dropout_prob:
                    keep_mask[i] = 0.0
            
            # Ensure at least one is kept (确保至少保留一个模态)
            if sum(keep_mask) == 0:
                keep_mask[random.randint(0, num_modalities - 1)] = 1.0
                
            # Apply mask (应用掩码)
            for i, feature in enumerate(features_list):
                final_weight = modality_gate[:, i].view(batch_size, 1, 1) * keep_mask[i]
                processed_features.append(feature * final_weight)
                
        else:
            # Inference or no dropout (推理阶段或未启用Dropout)
            for i, feature in enumerate(features_list):
                weight = modality_gate[:, i].view(batch_size, 1, 1)
                processed_features.append(feature * weight)

        # Expand fusion tokens (扩展融合 Token 以匹配批次大小)
        expanded_fusion_tokens = self.fusion_tokens.expand(batch_size, -1, -1)

        modality_tokens = torch.cat(processed_features, dim=1)
        fusion_tokens = expanded_fusion_tokens

        for layer_idx in range(self.num_layers):
            cross_norm = self.cross_norms[layer_idx](fusion_tokens)
            cross_out, _ = self.cross_attn_layers[layer_idx](
                query=cross_norm,
                key=modality_tokens,
                value=modality_tokens,
                need_weights=False,
            )
            fusion_tokens = fusion_tokens + cross_out

            self_norm = self.self_norms[layer_idx](fusion_tokens)
            self_out, _ = self.self_attn_layers[layer_idx](
                query=self_norm,
                key=self_norm,
                value=self_norm,
                need_weights=False,
            )
            fusion_tokens = fusion_tokens + self_out

            ffn_input = self.ffn_norms[layer_idx](fusion_tokens)
            fusion_tokens = fusion_tokens + self.ffn_layers[layer_idx](ffn_input)

        fused_bottleneck = fusion_tokens

        return fused_bottleneck