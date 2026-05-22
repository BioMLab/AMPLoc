import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class AgentInferenceManager:
    """
    AgentInferenceManager (控制循环)
    管理“直到最优”的迭代推理过程。

    逻辑说明：
    这个类是推理过程的指挥官。它不直接训练模型，而是利用训练好的 Agent 来动态调整
    各个模态（通道）的权重。
    核心流程是 predict_with_agent：
    1. 初始状态：所有通道权重为 1.0。
    2. 循环迭代：
       - 根据当前预测结果（置信度、熵）和上一步权重构建状态 (construct_state)。
       - Agent 根据状态输出新的权重 (get_action)。
       - 使用新权重重新融合并分类。
       - 如果置信度提高，则更新最佳结果。
       - 如果满足早停条件（置信度极高、性能下降、权重变化微小），则停止循环。
    """
    def __init__(self, model, agent, config):
        self.model = model
        self.agent = agent
        self.config = config
        
        self.episode_length = config['channel_agent']['episode_length']
        self.state_dim = config['channel_agent']['state_dim']
        self.fusion_dim = config['mbt_fuser']['fusion_dim']
        
        # 投影层：将融合嵌入压缩到状态维度
        self.fusion_proj = nn.Linear(self.fusion_dim, self.state_dim).to(model.device)
        
        # 活跃融合通道列表，用于将 Agent 输出映射到通道名称
        self.active_channels = config['meta_architect']['active_fusion_channels']
        num_channels = len(self.active_channels)
        
        # --- 任务 2: 修复 RPI 特征感知 (解耦) ---
        # 我们不再平均 RPI 特征，而是单独投影每个通道的特征
        # 并将它们连接到状态。这允许 Agent 看到每个通道的“指纹”。
        self.sig_dim = 16 # 每个通道签名的维度
        self.channel_projector = nn.Linear(self.fusion_dim, self.sig_dim).to(model.device)
        
        # 计算输入维度
        # 1 (熵) + state_dim (融合上下文) + num_channels (上一步权重)
        # [修改] 移除了通道签名以恢复基线稳定性
        input_dim = 1 + self.state_dim + num_channels 
        
        logger.info(f"Agent State Input Dim: {input_dim} (Entropy=1, Fusion={self.state_dim}, Weights={num_channels})")
        
        # 状态编码器：将 raw_state 投影到 state_dim
        self.state_encoder = nn.Linear(input_dim, self.state_dim).to(model.device)

    def construct_state(self, logits, fusion_emb, previous_weights, cached_features):
        """
        构建状态：
        构建代理的状态向量。
        """
        # 1. Logits 的熵 (衡量不确定性)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True) # [Batch, 1]
        
        # 2. 投影后的融合嵌入 (当前融合状态的上下文)
        proj_fusion = self.fusion_proj(fusion_emb) # [Batch, state_dim]
        
        # 3. 上一步的权重 (Agent 需要知道它之前的操作)
        # previous_weights: [Batch, num_channels]
        
        # 4. 通道签名 (已移除)
        # channel_sigs = []
        # ...
        
        # 连接所有组件 (熵, 融合上下文, 权重)
        raw_state = torch.cat([entropy, proj_fusion, previous_weights], dim=1)
        
        # 编码到 state_dim
        state = torch.relu(self.state_encoder(raw_state))
        return state
        
    def predict_with_agent(self, batch):
        """
        带代理预测：
        使用代理指导运行推理。
        
        参数:
            batch: 输入批次
            
        返回:
            best_logits: 循环过程中找到的最佳 Logits。
        """
        self.model.eval()
        self.agent.eval()
        
        # [修改] 如果配置中禁用了 Agent，则跳过循环并使用默认权重。
        if not self.config.get('channel_agent', {}).get('enabled', False):
            with torch.no_grad():
                cached_features = self.model.extract_all_features(batch)
                batch_size = list(cached_features.values())[0].size(0)
                num_channels = len(self.active_channels)
                current_weights_tensor = torch.ones(batch_size, num_channels, device=self.model.device)
                current_weights_dict = self._tensor_to_dict(current_weights_tensor)
                logits, _ = self.model.fusion_and_classify(cached_features, current_weights_dict)
                return logits

        with torch.no_grad():
            # 步骤 0 (缓存): 调用 model.extract_all_features(x)。
            cached_features = self.model.extract_all_features(batch)
            batch_size = list(cached_features.values())[0].size(0)
            
            # 步骤 1 (初始化): 初始化 current_weights 为全 1.0。
            # 权重形状: [Batch, NumChannels]
            num_channels = len(self.active_channels)
            current_weights_tensor = torch.ones(batch_size, num_channels, device=self.model.device)
            
            # 将张量权重转换为字典供模型使用
            current_weights_dict = self._tensor_to_dict(current_weights_tensor)
            
            # 运行 fusion_and_classify 获取 initial_logits 和 initial_fusion_emb。
            logits, fusion_emb = self.model.fusion_and_classify(cached_features, current_weights_dict)
            
            best_logits = logits
            best_confidence, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
            
            previous_weights = current_weights_tensor
            
            # 步骤 2 (循环): 迭代最多 config.episode_length 次
            for step in range(self.episode_length):
                # 使用辅助函数构建状态
                state = self.construct_state(logits, fusion_emb, previous_weights, cached_features)
                
                # Agent 动作: 将状态传递给 Agent。获取 new_weights。
                # Agent 返回 action_mean [Batch, num_channels]
                policy_bundle = self.agent.get_policy_bundle(state)
                new_weights_tensor = policy_bundle['action_mean']
                
                # [调试日志]
                # 打印第一个样本的权重
                # first_sample_weights = new_weights_tensor[0].detach().cpu().numpy()
                # print(f"[DEBUG Loop Step {step}] Agent generated weights (Sample 0): {first_sample_weights}")
                
                # 更新: 使用 new_weights 运行 fusion_and_classify。
                new_weights_dict = self._tensor_to_dict(new_weights_tensor)
                new_logits, new_fusion_emb = self.model.fusion_and_classify(
                    cached_features,
                    new_weights_dict,
                    policy_bundle=policy_bundle,
                )
                
                # 比较: 计算置信度 (例如, 最大概率)。
                # 如果 new_confidence > best_confidence, 更新 best_logits。
                new_confidence, _ = torch.max(torch.softmax(new_logits, dim=1), dim=1)
                
                # [调试日志]
                # 打印第一个样本的置信度变化
                # old_conf = best_confidence[0].item()
                # new_conf = new_confidence[0].item()
                # print(f"[DEBUG Loop Step {step}] Confidence (Sample 0): {old_conf:.4f} -> {new_conf:.4f}")
                
                # --- 智能耐心 & 振荡检查 ---
                
                # 1. 阈值早停
                # 如果置信度非常高，立即停止。
                # 我们检查批次中的平均置信度是否 > 0.98
                if new_confidence.mean() > 0.98:
                    # print(f"[DEBUG Loop Step {step}] Early Stopping: Mean confidence {new_confidence.mean():.4f} > 0.98")
                    # 最后一次更新 best logits
                    improved_mask = (new_confidence > best_confidence)
                    best_confidence = torch.where(improved_mask, new_confidence, best_confidence)
                    mask_expanded = improved_mask.unsqueeze(1).expand_as(best_logits)
                    best_logits = torch.where(mask_expanded, new_logits, best_logits)
                    break

                # 2. 振荡/退化检查
                # 如果置信度显著下降 (例如 -0.05)，恢复并停止。
                degradation_mask = (new_confidence < best_confidence - 0.05)
                if degradation_mask.any():
                    # print(f"[DEBUG Loop Step {step}] Degradation detected in {degradation_mask.sum().item()} samples. Reverting.")
                    # 如果平均退化严重，则中断循环
                    if degradation_mask.float().mean() > 0.2: # 如果 >20% 的样本退化
                         print(f"[DEBUG Loop Step {step}] Significant degradation. Stopping loop.")
                         break
                
                # 更新每个样本的 best_logits
                improved_mask = (new_confidence > best_confidence)
                
                # 我们需要在改进的地方更新 best_logits。
                # best_logits[improved_mask] = new_logits[improved_mask]
                # 但我们也需要跟踪 best_confidence
                best_confidence = torch.where(improved_mask, new_confidence, best_confidence)
                
                # 更新 best_logits 张量
                # 扩展 mask 以匹配 logits 维度
                mask_expanded = improved_mask.unsqueeze(1).expand_as(best_logits)
                best_logits = torch.where(mask_expanded, new_logits, best_logits)
                
                # 早停: 如果权重变化非常小，中断循环。
                # 我们检查批次的平均变化
                weight_change = torch.mean(torch.abs(new_weights_tensor - previous_weights))
                if weight_change < 1e-3:
                    break
                
                # 为下一次迭代更新
                # Update for next iteration
                logits = new_logits
                fusion_emb = new_fusion_emb
                previous_weights = new_weights_tensor
                
            return best_logits

    def _tensor_to_dict(self, weights_tensor):
        """
        将权重张量 [Batch, NumChannels] 转换为字典 {channel_name: tensor [Batch, 1, 1]}
        """
        weights_dict = {}
        for i, channel_name in enumerate(self.active_channels):
            # 提取该通道的列
            w = weights_tensor[:, i] # [Batch]
            # 重塑为 [Batch, 1, 1] 以进行广播
            w = w.view(-1, 1, 1)
            weights_dict[channel_name] = w
        return weights_dict
