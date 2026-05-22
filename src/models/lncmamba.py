import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm is not installed. Wordv2ec and Mamba model will not work.")
    Mamba = nn.Identity

logger = logging.getLogger(__name__)


class LncMamba(nn.Module):
    """
   基于 Mamba LncRNA序列特征提取器。
    
    功能：
    提取 LncRNA 序列的局部特征 (通过 Wordv2ec)。
    提取 LncRNA 序列的长程依赖特征 (Mamba)。
    使用特定于类别的注意力机制 (Class-Specific Attention) 生成分类特征。
    支持 Motif 增强机制，提高模型对关键序列片段的关注度。
    """
    def __init__(self, num_classes, vocab_size, lncmamba_cfg, fusion_cfg):
        """
        初始化 Mamba。
        
        Args:
            num_classes (int): 分类类别数。
            vocab_size (int): 词汇表大小 (k-mer 种类数)。
            lncmamba_cfg (dict): LncMamba 配置字典。
            fusion_cfg (dict): 融合配置字典，用于确定输出投影维度。
        """
        super().__init__()
        embedding_dim = lncmamba_cfg['embedding_dim']
        mamba_d_state = lncmamba_cfg['mamba_d_state']
        mamba_d_conv = lncmamba_cfg['mamba_d_conv']
        mamba_expand = lncmamba_cfg['mamba_expand']
        # 使用 .get() 确保了鲁棒性
        self.use_motif_enhancement = lncmamba_cfg.get('use_motif_enhancement', False)
        self.motif_tkn_ids = lncmamba_cfg.get('motif_tkn_ids', [])
        self.motif_factor = lncmamba_cfg.get('motif_factor', 1.5)

        fusion_dim = fusion_cfg['fusion_dim']

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 局部特征提取层 (不同卷积核大小捕获不同尺度的局部模式)
        self.conv1_k3 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2_k5 = nn.Conv1d(embedding_dim, 64, kernel_size=5, padding=2)

        # Mamba 输入维度 = 嵌入维度 + 提取到的特征的维度
        mamba_input_dim = embedding_dim + 128 + 64  # Should be 704
        
        # Mamba 核心模块 (状态空间模型)
        self.mamba = Mamba(d_model=mamba_input_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)

        self.num_classes = num_classes
        
        # 辅助分类头 (Auxiliary Head)
        # 为每个类别学习独立的注意力权重和输出权重
        self.aux_head = nn.ModuleDict({
            'attention_weights_W': nn.ParameterList(
                [nn.Parameter(torch.randn(mamba_input_dim, 1)) for _ in range(num_classes)]),
            'attention_weights_b': nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_classes)]),
            'output_weights_W': nn.ParameterList(
                [nn.Parameter(torch.randn(mamba_input_dim, 1)) for _ in range(num_classes)]),
            'output_weights_b': nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_classes)])
        })

        # 特征投影层：将 Mamba 输出投影到统一的融合维度
        self.projection = nn.Linear(mamba_input_dim, fusion_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.aux_head['attention_weights_W']: nn.init.xavier_uniform_(param)
        for param in self.aux_head['output_weights_W']: nn.init.xavier_uniform_(param)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播。
        
        Args:
            input_ids: 输入序列 Token ID [B, L]。
            attention_mask: 注意力掩码 [B, L]。
            
        Returns:
            dict: 包含 'logits' (辅助分类结果) 和 'features' (投影后的序列特征)。
        """
        # 1. 嵌入和 CNN 特征提取
        x_embed = self.embedding(input_ids)
        x_conv_input = x_embed.permute(0, 2, 1)

        feat_conv1_k3 = F.relu(self.conv1_k3(x_conv_input)).permute(0, 2, 1)
        feat_conv2_k5 = F.relu(self.conv2_k5(x_conv_input)).permute(0, 2, 1)

        # 2. 特征拼接
        features = torch.cat([x_embed, feat_conv1_k3, feat_conv2_k5], dim=2)
        
        # 3. Mamba 序列建模
        features = self.mamba(features)

        batch_size = features.shape[0]
        final_logits_list = []

        # 4. 逐类别注意力机制 (Class-Specific Attention)
        #是指模型在尝试为每一个可能的分类目标（类别）寻找最相关的特征。
        for i in range(self.num_classes):
            # 计算注意力分数
            attn_logits = features @ self.aux_head['attention_weights_W'][i] + self.aux_head['attention_weights_b'][i]

            if attention_mask is not None:
                attn_logits = attn_logits.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))

            attn_weights = F.softmax(attn_logits, dim=1)

            # Motif 增强机制：如果启用了 Motif 增强，增加特定 Motif Token 的注意力权重
            if self.use_motif_enhancement and self.motif_tkn_ids:
                motif_multiplier = torch.ones_like(attn_weights, device=attn_weights.device)
                for batch_idx in range(batch_size):
                    seq_tokens = input_ids[batch_idx]
                    for motif_id in self.motif_tkn_ids:
                        motif_indices = (seq_tokens == motif_id).nonzero(as_tuple=True)[0]
                        if motif_indices.numel() > 0:
                            motif_multiplier[batch_idx, motif_indices, :] *= self.motif_factor
                attn_weights = attn_weights * motif_multiplier

            # 计算上下文向量 (加权平均)
            context_vector = torch.bmm(attn_weights.transpose(1, 2), features)

            # 计算该类别的 Logit
            class_logit = context_vector @ self.aux_head['output_weights_W'][i] + self.aux_head['output_weights_b'][i]

            # 维度调整
            if class_logit.dim() == 3:
                class_logit_squeezed = class_logit.view(batch_size)
            else:
                class_logit_squeezed = class_logit.squeeze()
            
            if class_logit_squeezed.dim() == 1:
                class_logit_squeezed = class_logit_squeezed.unsqueeze(1)
            
            final_logits_list.append(class_logit_squeezed)

        # 拼接所有类别的 Logits
        aux_logits = torch.cat(final_logits_list, dim=1)

        # 5. 特征投影 (用于后续融合)
        projected_features = self.projection(features)

        return {
            "logits": aux_logits,
            "features": projected_features
        }
