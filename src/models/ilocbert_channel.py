# src/models/ilocbert_channel.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import logging

logger = logging.getLogger(__name__)


class ILocBertChannel(nn.Module):
  
    def __init__(self, config, num_classes, fusion_cfg):
        super().__init__()
        self.config = config
        fusion_dim = fusion_cfg['fusion_dim']

        logger.info("Initializing ILocBertChannel with DNABERT-2...")
        
        # 【关键修改】临时屏蔽 triton，强制 DNABERT-2 使用 PyTorch 原生 Attention
        # 这是为了解决 triton 版本不兼容导致的 "dot() got an unexpected keyword argument 'trans_b'" 错误
        import sys
        original_triton = sys.modules.get('triton')
        sys.modules['triton'] = None

        try:
            # 加载配置
            bert_config = AutoConfig.from_pretrained(
                "pretrained/DNABERT-2-117M",
                trust_remote_code=True
            )
            
            # 加载预训练的 DNABERT-2 模型
            self.bert_model = AutoModel.from_pretrained(
                "pretrained/DNABERT-2-117M",
                config=bert_config,
                trust_remote_code=True,
                add_pooling_layer=False,  
            )
            logger.info("Successfully loaded DNABERT-2 (Triton disabled).")

        except Exception as e:
            logger.error(f"Failed to load DNABERT-2 model from local path. Error: {e}")
            raise e
        finally:
            # 恢复 triton (虽然本项目其他部分可能不用，但保持环境清洁是好习惯)
            if original_triton is not None:
                sys.modules['triton'] = original_triton
            else:
                del sys.modules['triton']

        # 冻结部分BERT层以加速训练并防止灾难性遗忘 
        # for param in self.bert_model.encoder.layer[:6].parameters():
        #     param.requires_grad = False
        # logger.info("Froze the first 6 layers of DNABERT-2.")

        # 定义辅助分类头
        self.aux_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

        # 定义用于MBT融合的投影层
        self.projection = nn.Linear(768, fusion_dim)

        logger.info(f"ILocBertChannel initialized. Output features will be projected to {fusion_dim} for fusion.")

    def forward(self, input_ids, attention_mask):
        '''
        Args:
            input_ids (torch.Tensor): Token IDs from DNABERT-2 tokenizer.
            attention_mask (torch.Tensor): Attention mask.
        Returns:
            dict: A dictionary containing 'logits' and 'features'.
        '''
        # 获取BERT模型的输出
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
            
        pooled_output = self.pool_hidden_state(last_hidden_state, attention_mask)  # Shape: [batch_size, 768]

        # 计算辅助 logits
        aux_logits = self.aux_head(pooled_output)

        # 投影特征以用于融合
        projected_features = self.projection(pooled_output)

        # BERT模型通常产生一个序列的特征，为匹配其他通道，我们增加一个伪序列维度
        # Shape: [batch_size, 1, fusion_dim]
        projected_features = projected_features.unsqueeze(1)

        return {
            "logits": aux_logits,
            "features": projected_features
        }

    def pool_hidden_state(self, last_hidden_state, attention_mask):
        # 为了精确地进行均值池化，我们只考虑非padding部分
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
  
