import torch
import torch.nn as nn
from torch.distributions import Normal

class PPOAgent(nn.Module):
    """
    PPOAgent (大脑)
    一个简单的 Actor-Critic 网络，用于动态通道选择和融合权重分配。

    逻辑说明：
    这个类实现了强化学习中的 PPO (Proximal Policy Optimization) 算法的核心网络结构。
    它包含两个部分：
    1. Actor (策略网络): 接收状态，输出动作（即每个通道的权重）。
       - 使用 Sigmoid 激活函数，确保输出在 [0, 1] 之间。
       - 最后一层偏置初始化为 2.0，使得初始输出接近 0.88，即一开始倾向于使用所有通道（保守策略）。
    2. Critic (价值网络): 接收状态，输出当前状态的价值估计 (Value)，用于训练时的优势函数计算。
    """
    def __init__(self, state_dim, num_channels, hidden_dim=128):
        super(PPOAgent, self).__init__()
        
        # Actor 网络
        # 输入: state (形状: [Batch, state_dim])
        # 输出: action_mean (形状: [Batch, num_channels])
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )
        
        # 关键技术细节：将 Actor 最后一层的偏置初始化为正值。
        # 这确保代理从“保守策略”（高权重/所有通道激活）开始。
        # Sigmoid(2.0) ~= 0.88
        # nn.init.constant_(self.actor[-1].bias, 2.0)
        
        # [修改] 初始化偏置为 2.0 (Sigmoid(2.0) ~= 0.88) 以恢复保守策略（高初始权重）。
        nn.init.constant_(self.actor[-1].bias, 2.0)

        # Mask 分支：学习 sample-specific 的通道启用概率
        self.mask_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels)
        )
        nn.init.constant_(self.mask_head[-1].bias, 1.5)

        # H-params 分支：输出样本级的融合调制参数
        # 约定顺序：temperature, selection_bias, sparsity_gate
        self.hparam_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        nn.init.constant_(self.hparam_head[-1].bias, 0.0)
        
        # Critic 网络
        # 输入: state (形状: [Batch, state_dim])
        # 输出: value (标量)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        代理的前向传播。
        
        参数:
            state: 形状为 [Batch, state_dim] 的张量
            
        返回:
            action_mean: 形状为 [Batch, num_channels] 的张量 (值在 [0, 1] 之间)
            value: 形状为 [Batch, 1] 的张量
        """
        policy = self._build_policy(state)
        return policy["action_mean"], policy["value"]

    def _build_policy(self, state):
        """构建完整策略分支：weight / mask / hparams。"""
        weight_logits = self.actor(state)
        weight_mean = torch.sigmoid(weight_logits)

        mask_logits = self.mask_head(state)
        mask_prob = torch.sigmoid(mask_logits)

        hparam_logits = self.hparam_head(state)
        temperature = 0.75 + 0.75 * torch.sigmoid(hparam_logits[:, 0:1])
        selection_bias = 0.25 + 0.50 * torch.sigmoid(hparam_logits[:, 1:2])
        sparsity_gate = torch.sigmoid(hparam_logits[:, 2:3])

        # 先用 mask 产生样本级启用概率，再用 temperature 做软调制。
        mask_gate = torch.sigmoid((mask_logits - selection_bias) / temperature)
        blended_gate = sparsity_gate * mask_gate + (1.0 - sparsity_gate) * mask_prob

        action_logits = weight_logits * blended_gate
        action_mean = torch.sigmoid(action_logits / temperature)

        value = self.critic(state)

        return {
            "action_mean": action_mean,
            "weight_mean": weight_mean,
            "mask_prob": mask_prob,
            "effective_mask": blended_gate,
            "hparams": {
                "temperature": temperature,
                "selection_bias": selection_bias,
                "sparsity_gate": sparsity_gate,
            },
            "value": value,
        }

    def get_policy_bundle(self, state):
        """返回完整策略分支，便于训练/推理阶段做更细粒度的调制。"""
        return self._build_policy(state)
    
    def get_action(self, state, deterministic=False):
        """
        从状态获取动作。
        
        参数:
            state: [Batch, state_dim]
            deterministic: 如果为 True，返回均值。如果为 False，进行采样（此处简单的 Sigmoid 未完全实现采样，
                           通常 PPO 对连续动作使用正态分布）。
                           
        针对此特定提示要求“Actor 输出：action_mean ... 使用 Sigmoid 输出激活”，
        我们将输出直接视为动作或分布参数。
        如果我们需要探索，可能会添加噪声。
        """
        action_mean, value = self(state)
        
        if deterministic:
            return action_mean, value
        else:
            # 探索策略：如果需要，在训练期间添加小的高斯噪声
            # 遵循专注于架构的提示。
           
            return action_mean, value
