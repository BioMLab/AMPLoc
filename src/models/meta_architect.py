import torch
import torch.nn as nn
import logging
import os
from .lncmamba import LncMamba
from .rnaloclm_channel import RNALocLMChannel
from .cfploc_channel import CFPLncLocChannel
# from .mbt_fuser import MBT_Fuser  # [# MBT-MOD] Import the new fuser
from .robust_mbt_fuser import RobustMBT_Fuser as MBT_Fuser # Use Robust Fuser
from .ilocbert_channel import ILocBertChannel
from .intra_graph_channel import IntraGraphChannel
from .rpi_channel import RPIChannel

logger = logging.getLogger(__name__)


class MetaArchitect(nn.Module):
    """
    MetaArchitect: LncAPNet 的核心元架构类。
    
    功能：
    1. 负责初始化和管理所有子通道（LncMamba, RNALocLM, CFPLncLoc, iLocBERT, IntraGraph, RPI）。
    2. 负责调用多模态融合器（MBT_Fuser）将各通道特征进行融合。
    """
    def __init__(self, config, tokenizer, motif_tkn_ids, device, project_root):
        """
        初始化 MetaArchitect。
        
        Args:
            config (dict): 全局配置字典，包含各通道和融合器的参数。
            tokenizer (Tokenizer): 用于获取词汇表大小和类别数。
            motif_tkn_ids (list): 用于 LncMamba 通道的 Motif Token ID 列表。
            device (torch.device): 运行设备 (CPU/GPU)。
            project_root (str): 项目根目录路径，用于定位数据文件。
        """
        super().__init__()
        self.config = config
        self.device = device
        self.channels = nn.ModuleDict()
        num_classes = tokenizer.labNum

        # [# MBT-MOD] Get fusion config once
        fusion_cfg = config['mbt_fuser']
        fusion_dim = fusion_cfg['fusion_dim']

        # --- Initialize Channels (now with fusion_cfg) ---
        # 初始化 LncMamba 通道 (序列模态)
        if config.get('lncmamba', {}).get('enabled', True):
            logger.info("Initializing Channel 1: LncMamba for MBT")
            self.channels['lncmamba'] = LncMamba(
                num_classes=num_classes,
                vocab_size=tokenizer.tknNum,
                lncmamba_cfg={**config['lncmamba'], 'motif_tkn_ids': motif_tkn_ids},
                fusion_cfg=fusion_cfg
            )

        # 初始化 RNALoc-FM 通道 (预训练语言模型模态)
        if config.get('rnaloclm', {}).get('enabled', False):
            logger.info("Initializing Channel 2: RNALoc-LM for MBT")
            self.channels['rnaloclm'] = RNALocLMChannel(
                config=config['rnaloclm'],
                num_classes=num_classes,
                device=self.device,
                fusion_cfg=fusion_cfg
            )

        # 初始化 CFPLncLoc 通道 (CGR图像模态)
        if config.get('cfploc', {}).get('enabled', False):
            logger.info("Initializing Channel 3: CFPLncLoc for MBT")
            self.channels['cfploc'] = CFPLncLocChannel(
                config=config['cfploc'],
                num_classes=num_classes,
                fusion_cfg=fusion_cfg
            )
        
        # 初始化 iLoc-BERT 通道 (BERT特征模态)
        if config.get('ilocbert', {}).get('enabled', False):
            logger.info("Initializing Channel 4: iLoc-BERT for MBT")
            self.channels['ilocbert'] = ILocBertChannel(
                config=config['ilocbert'],
                num_classes=num_classes,
                fusion_cfg=fusion_cfg
            )

        # 初始化Intra-Graph 通道 (RNA图结构模态)
        if config.get('intra_graph_channel', {}).get('enabled', False):
            logger.info("Initializing Channel 5: Intra-Graph (GIN) for MBT")
            self.channels['intra_graph_channel'] = IntraGraphChannel(
                config=config, num_classes=num_classes, fusion_cfg=fusion_cfg
            )

        # 初始化 RPI 通道 (RNA-蛋白质相互作用模态)
        if config.get('rpi_channel', {}).get('enabled', False):
            logger.info("Initializing Channel 7: RPI (Guided Attention) for MBT")
            self.channels['rpi_channel'] = RPIChannel(
                config=config['rpi_channel'],
                num_classes=num_classes,
                fusion_cfg=fusion_cfg
            )

        # [Modification] Token Compressor for LncMamba
        # To balance the number of tokens between LncMamba (64) and RPI (22),
        # we add a compressor to reduce LncMamba tokens to 22.
        self.lncmamba_compressor = None
        self.rnaloclm_compressor = None
        self.cfploc_compressor = None
        self.ilocbert_compressor = None
        self.intra_graph_compressor = None

        if 'rpi_channel' in self.channels:
             # Target length: RPI seq_len (22)
             target_len = config['rpi_channel'].get('seq_len', 22)
             
             if 'lncmamba' in self.channels:
                 self.lncmamba_compressor = nn.AdaptiveAvgPool1d(target_len)
                 logger.info(f"Initialized LncMamba Token Compressor: -> {target_len}")
            
             if 'rnaloclm' in self.channels:
                 self.rnaloclm_compressor = nn.AdaptiveAvgPool1d(target_len)
                 logger.info(f"Initialized RNALocLM Token Compressor: -> {target_len}")

             if 'cfploc' in self.channels:
                 self.cfploc_compressor = nn.AdaptiveAvgPool1d(target_len)
                 logger.info(f"Initialized CFPLncLoc Token Compressor: -> {target_len}")

             if 'ilocbert' in self.channels:
                 self.ilocbert_compressor = nn.AdaptiveAvgPool1d(target_len)
                 logger.info(f"Initialized iLocBERT Token Compressor: -> {target_len}")

             if 'intra_graph_channel' in self.channels:
                 self.intra_graph_compressor = nn.AdaptiveAvgPool1d(target_len)
                 logger.info(f"Initialized Intra-Graph Token Compressor: -> {target_len}")

        if not self.channels:
            raise ValueError("No channels are enabled in the configuration.")
        logger.info(f"Enabled channels: {list(self.channels.keys())}")

        # --- Initialize the MBT Fuser ---
        # 初始化多模态瓶颈变换器 (MBT) 融合模块
        self.fuser = MBT_Fuser(fusion_cfg)
        logger.info("MBT Fuser Initialized.")

        # --- Initialize Final Classifier ---
        # 初始化最终分类器：接收融合后的特征，输出最终预测
        # It takes the pooled output of the fuser's bottleneck tokens
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )
        logger.info("Final Classifier Initialized.")

        self.to(device)

    def forward(self, batch):
        """
        前向传播函数。
        
        Args:
            batch (dict): 包含各种输入数据的批次字典。
            
        Returns:
            dict: 包含 'fused_logits' (最终预测) 和 'channel_logits' (各通道辅助预测) 的结果字典。
        """
        channel_outputs = {}

        # --- 1. Get features and auxiliary logits from all enabled channels ---
        # 依次调用各个启用的通道，获取特征和辅助logits
        
        if 'lncmamba' in self.channels:
            channel_outputs['lncmamba'] = self.channels['lncmamba'](
                batch['input_ids'].to(self.device),
                batch['attention_mask'].to(self.device)
            )

        if 'rnaloclm' in self.channels:
            channel_outputs['rnaloclm'] = self.channels['rnaloclm'](batch['raw_sequences'])

        if 'cfploc' in self.channels:
            channel_outputs['cfploc'] = self.channels['cfploc'](batch['cgr_features'].to(self.device))

        if 'ilocbert' in self.channels:
            channel_outputs['ilocbert'] = self.channels['ilocbert'](
                batch['iloc_input_ids'].to(self.device),
                batch['iloc_attention_mask'].to(self.device)
            )

        # 调用Intra-Graph 通道
        if 'intra_graph_channel' in self.channels:
            if batch.get('graph_data') is not None:
                channel_outputs['intra_graph_channel'] = self.channels['intra_graph_channel'](
                    batch['graph_data'].to(self.device)
                )
            else:
                # 如果整个批次都没有图数据，或graph_data不符合预期，我们不向channel_outputs添加任何东西。
                # 后续的 `features_to_fuse` 列表将只包含其他通道的输出，逻辑依然正确。
                logger.warning(
                    "Skipping intra_graph_channel for this batch as no valid graph data was found.")

        if 'rpi_channel' in self.channels:
            # 从批次中获取gene_ids
            gene_ids = batch.get('gene_ids')
            if gene_ids:
                # 检查是否需要返回注意力权重（用于分析）
                return_attention = batch.get('return_attention', False)
                # 调用rpi_channel，它返回特征和logits
                rpi_output = self.channels['rpi_channel'](gene_ids, return_attention=return_attention)
                channel_outputs['rpi_channel'] = rpi_output
            else:
                logger.warning("Skipping rpi_channel as 'gene_ids' not found in batch.")


        # --- 2. Prepare for fusion and collect auxiliary logits ---
        # 准备融合所需的特征列表
        features_to_fuse = []
        channel_names_for_fusion = []
        channel_logits = {}

        for channel_name, out in channel_outputs.items():
            if out is None: # Handle None return from channel
                continue
                
            features = out['features']
            logits = out.get('logits') # Logits might not be in every output structure?
            
            if isinstance(features, dict):
                for sub_name, feat in features.items():
                    if feat.dim() == 2:
                        feat = feat.unsqueeze(1)
                    features_to_fuse.append(feat)
                    channel_names_for_fusion.append(sub_name)

                if isinstance(logits, dict):
                    for sub_name, logit in logits.items():
                        channel_logits[sub_name] = logit
                elif logits is not None:
                    channel_logits[channel_name] = logits
                continue

            if logits is not None:
                channel_logits[channel_name] = logits

            if features.dim() == 2:
                # 如果是 [B, D] 格式，转换为 [B, 1, D]
                features = features.unsqueeze(1)
            elif features.dim() == 3:
                # 如果已经是 [B, seq_len, D] 格式，保持不变
                pass
            else:
                logger.warning(f"Unexpected feature dimension for {channel_name}: {features.shape}")
                continue
            
            features_to_fuse.append(features)
            channel_names_for_fusion.append(channel_name)

        #features_to_fuse = [out['features'] for out in channel_outputs.values()]
        #channel_logits = {name: out['logits'] for name, out in channel_outputs.items()}

        if not features_to_fuse:
            # 这种情况很罕见，但需要处理。返回一个零张量的logits。
            batch_size = next(iter(batch.values())).size(0)
            num_classes = self.final_classifier[-1].out_features
            zeros_logits = torch.zeros(batch_size, num_classes, device=self.device)
            logger.error("No features were generated by any channel for this batch. Returning zero logits.")
            return {
                "fused_logits": zeros_logits,
                "channel_logits": {}
            }

        # If only one channel is active, bypass fusion
        # 如果只有一个通道被激活，则跳过融合步骤，直接使用该通道的输出
        if len(features_to_fuse) <= 1:
            result = {
                "fused_logits": list(channel_logits.values())[0],
                "channel_logits": channel_logits
            }
            # 如果需要返回注意力权重（用于分析），也返回channel_outputs
            if batch.get('return_attention', False):
                result["channel_outputs"] = channel_outputs
            return result

        # --- 3. Fuse features using MBT Fuser ---
        # 使用 MBT 融合器融合多模态特征
        # 获取通道权重配置
        channel_weights = self.config.get('meta_architect', {}).get('channel_weights', {})
        fused_bottleneck = self.fuser(features_to_fuse, channel_names_for_fusion, channel_weights)

        # --- 4. Classify based on fused representation ---
        # 基于融合后的特征进行最终分类
        # Pool the bottleneck tokens (mean pooling is robust)
        pooled_fused_features = torch.mean(fused_bottleneck, dim=1)
        fused_logits = self.final_classifier(pooled_fused_features)

        result = {
            "fused_logits": fused_logits,
            "channel_logits": channel_logits  # For auxiliary loss calculation in the trainer
        }
        
        # 如果需要返回注意力权重（用于分析），也返回channel_outputs
        if batch.get('return_attention', False):
            result["channel_outputs"] = channel_outputs
        
        return result
