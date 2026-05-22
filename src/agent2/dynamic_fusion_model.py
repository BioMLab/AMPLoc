import torch
import torch.nn as nn
import logging
from src.models.meta_architect import MetaArchitect

logger = logging.getLogger(__name__)

class DynamicFusionModel(MetaArchitect):
    """
    DynamicFusionModel (现有模型的修改版)
    继承自 MetaArchitect，支持动态融合权重和特征缓存。

    逻辑说明：
    这个类主要负责两个任务：
    1. extract_all_features: 一次性提取所有模态（通道）的特征，用于后续的快速迭代。
    2. fusion_and_classify: 根据 Agent 给出的权重，对特征进行加权、融合，并进行最终分类。
    """
    def __init__(self, config, tokenizer, motif_tkn_ids, device, project_root):
        super().__init__(config, tokenizer, motif_tkn_ids, device, project_root)
        
    def extract_all_features(self, batch):
        """Extract features for all enabled channels and flatten multi-view outputs."""
        logger.info(f"--- [Detailed Check] Start Processing Batch of size {len(batch.get('labels', []))} ---")
        channel_outputs = {}

        if 'lncmamba' in self.channels:
            logger.info("  > Channel: [lncmamba] - Starting...")
            out = self.channels['lncmamba'](
                batch['input_ids'].to(self.device),
                batch['attention_mask'].to(self.device)
            )
            feat = out['features']
            logger.info(f"  > Channel: [lncmamba] - Raw Output Shape: {feat.shape}")
            if hasattr(self, 'lncmamba_compressor') and self.lncmamba_compressor is not None:
                feat = feat.transpose(1, 2)
                feat = self.lncmamba_compressor(feat)
                feat = feat.transpose(1, 2)
                logger.info(f"  > Channel: [lncmamba] - Compressed Shape: {feat.shape}")
            channel_outputs['lncmamba'] = {'features': feat, 'logits': out.get('logits')}

        if 'rnaloclm' in self.channels:
            logger.info("  > Channel: [rnaloclm] - Starting...")
            out = self.channels['rnaloclm'](batch['raw_sequences'])
            feat = out['features']
            logger.info(f"  > Channel: [rnaloclm] - Raw Output Shape: {feat.shape}")
            if hasattr(self, 'rnaloclm_compressor') and self.rnaloclm_compressor is not None:
                feat = feat.transpose(1, 2)
                feat = self.rnaloclm_compressor(feat)
                feat = feat.transpose(1, 2)
            channel_outputs['rnaloclm'] = {'features': feat, 'logits': out.get('logits')}

        if 'cfploc' in self.channels:
            logger.info("  > Channel: [cfploc] - Starting...")
            out = self.channels['cfploc'](batch['cgr_features'].to(self.device))
            feat = out['features']
            logger.info(f"  > Channel: [cfploc] - Raw Output Shape: {feat.shape}")
            if hasattr(self, 'cfploc_compressor') and self.cfploc_compressor is not None:
                feat = feat.transpose(1, 2)
                feat = self.cfploc_compressor(feat)
                feat = feat.transpose(1, 2)
            channel_outputs['cfploc'] = {'features': feat, 'logits': out.get('logits')}

        if 'ilocbert' in self.channels:
            logger.info("  > Channel: [ilocbert] - Starting...")
            try:
                out = self.channels['ilocbert'](
                    batch['iloc_input_ids'].to(self.device),
                    batch['iloc_attention_mask'].to(self.device)
                )
                feat = out['features']
                logger.info(f"  > Channel: [ilocbert] - Raw Output Shape: {feat.shape}")
                if hasattr(self, 'ilocbert_compressor') and self.ilocbert_compressor is not None:
                    feat = feat.transpose(1, 2)
                    feat = self.ilocbert_compressor(feat)
                    feat = feat.transpose(1, 2)
                channel_outputs['ilocbert'] = {'features': feat, 'logits': out.get('logits')}
            except Exception as e:
                logger.error(f"  > Channel: [ilocbert] - FAILED: {e}")
                raise e

        if 'intra_graph_channel' in self.channels:
            logger.info("  > Channel: [intra_graph_channel] - Starting...")
            if batch.get('graph_data') is not None:
                channel_outputs['intra_graph_channel'] = self.channels['intra_graph_channel'](
                    batch['graph_data'].to(self.device)
                )
            else:
                logger.warning("  > Channel: [intra_graph_channel] - No valid graph data found.")

        if 'rpi_channel' in self.channels:
            logger.info("  > Channel: [rpi_channel] - Starting...")
            gene_ids = batch.get('gene_ids')
            if gene_ids:
                channel_outputs['rpi_channel'] = self.channels['rpi_channel'](gene_ids)

        logger.info(f"--- [Detailed Check] Batch Processing Complete. Extracted {len(channel_outputs)} channel groups. ---")

        processed_outputs = {}
        for name, out in channel_outputs.items():
            if out is None:
                continue
            features = out['features']
            if isinstance(features, dict):
                for sub_name, feat in features.items():
                    if feat.dim() == 2:
                        feat = feat.unsqueeze(1)
                    processed_outputs[sub_name] = feat
            else:
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                processed_outputs[name] = features

        return processed_outputs

    def fusion_and_classify(self, cached_features, channel_weights, policy_bundle=None):
        """
        融合与分类：
        输入预计算的特征字典和标量权重字典（来自 Agent）。
        
        参数:
            cached_features: Dict[str, Tensor] - extract_all_features 的输出
            channel_weights: Dict[str, Tensor] - 每个通道的权重。
                             Tensor 形状应可广播到特征形状，例如 [Batch, 1, 1]
        """
        # [DEBUG LOGGING]
        # if channel_weights:
        #     # Calculate average weight for debug
        #     total_w = 0
        #     count_w = 0
        #     debug_weights = {}
        #     for k, v in channel_weights.items():
        #         mean_val = v.mean().item()
        #         total_w += mean_val
        #         count_w += 1
        #         debug_weights[k] = f"{mean_val:.4f}"
            
        #     avg_w = total_w / max(count_w, 1)
        #     print(f"[DEBUG Model] Running fusion. Avg Weight: {avg_w:.4f}")
        #     print(f"[DEBUG Model] Individual Weights: {debug_weights}")
        # else:
        #     print("[DEBUG Model] Running fusion with default/empty weights.")

        features_to_fuse = []
        channel_names_for_fusion = []
        
        # Iterate through cached features
        for channel_name, features in cached_features.items():
            # Keep raw features here and let the fuser apply the policy-guided modulation.
            # This keeps the modulation inside the fusion stage instead of doing a second
            # pre-multiplication before the bottleneck transformer.
            features_to_fuse.append(features)
            channel_names_for_fusion.append(channel_name)

        if not features_to_fuse:
             # Return dummy
            batch_size = list(cached_features.values())[0].size(0)
            num_classes = self.final_classifier[-1].out_features
            fusion_dim = self.fuser.fusion_tokens.size(-1)
            return torch.zeros(batch_size, num_classes, device=self.device), torch.zeros(batch_size, fusion_dim, device=self.device)

        # Pass the raw features and the policy weights into the bottleneck fuser.
        # The fuser performs the effective modulation inside the fusion stage.
        fused_bottleneck = self.fuser(
            features_to_fuse,
            channel_names_for_fusion,
            channel_weights,
            policy_bundle=policy_bundle,
        )

        # Pass fused result into Classifier
        # Pool the bottleneck tokens
        pooled_fused_features = torch.mean(fused_bottleneck, dim=1)
        logits = self.final_classifier(pooled_fused_features)
        
        # Output: logits, fusion_embedding (global pooled vector for state construction)
        return logits, pooled_fused_features
