import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.agent2.dynamic_fusion_model import DynamicFusionModel
from src.agent2.ppo_agent import PPOAgent
from src.agent2.agent_inference_manager import AgentInferenceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LncAPNetTrainer:
    """
    LncAPNetTrainer with Supervised and PPO Phases.
    Handles the training loop alternating between supervised training of the model
    and PPO reinforcement learning of the agent.
    """
    def __init__(self, model, agent, inference_manager, train_loader, val_loader, config):
        self.model = model
        self.agent = agent
        self.inference_manager = inference_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = model.device
        
        # Optimizers
        # Model optimizer: Trains the fusion model and backbones
        self.model_optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(config['training'].get('learning_rate', 2e-5))
        )
        
        # Agent optimizer: Trains the Agent AND the Manager's projection layers
        # We need to include inference_manager.state_encoder and inference_manager.fusion_proj
        agent_params = list(self.agent.parameters())
        if hasattr(self.inference_manager, 'state_encoder'):
            agent_params += list(self.inference_manager.state_encoder.parameters())
        if hasattr(self.inference_manager, 'fusion_proj'):
            agent_params += list(self.inference_manager.fusion_proj.parameters())
            
        self.agent_optimizer = optim.Adam(
            agent_params,
            lr=float(config['channel_agent'].get('learning_rate', 3e-4))
        )
        
        # [Modification] Use BCEWithLogitsLoss for Multi-Label Classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.active_channels = config['meta_architect']['active_fusion_channels']
        
        # PPO Hyperparameters
        self.clip_epsilon = config['channel_agent'].get('clip_epsilon', 0.2)
        self.value_coef = config['channel_agent'].get('value_coef', 0.5)
        self.exploration_std = 0.1 # Standard deviation for exploration noise

    def train_supervised_epoch(self, epoch):
        """
        Phase 1: Supervised Training Method
        Goal: Train the fusion model and backbones.
        """
        self.model.train()
        self.agent.eval() # Agent is not trained here
        
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Supervised Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            # Note: The model handles moving specific tensors, but we might need to ensure labels are on device
            labels = batch['labels'].to(self.device)
            
            # [Modification] For Multi-Label (BCE), we use float labels directly.
            targets = labels.float()

            self.model_optimizer.zero_grad()
            
            # Crucial: Do not use the Agent's output here. 
            # Create a "weights" dictionary where all weights are set to 1.0
            batch_size = labels.size(0)
            weights_dict = self._get_fixed_weights(batch_size, 1.0)
            
            # We need to extract features first or call a method that does it all?
            # DynamicFusionModel has extract_all_features and fusion_and_classify.
            # For supervised training, we can just run them in sequence.
            # Since we want to train backbones, we should NOT use torch.no_grad() for extraction.
            # But extract_all_features in DynamicFusionModel just calls sub-modules.
            
            # However, extract_all_features was designed for caching (maybe no_grad?).
            # Let's check DynamicFusionModel.extract_all_features implementation.
            # It just calls self.channels['name'](input). So it supports grad if context allows.
            
            features = self.model.extract_all_features(batch)
            logits, _ = self.model.fusion_and_classify(features, weights_dict)
            
            loss = self.criterion(logits, targets)
            loss.backward()
            self.model_optimizer.step()
            
            total_loss += loss.item()
            
            # [Modification] Calculate accuracy for multi-label
            predicted = (torch.sigmoid(logits) > 0.5).float()
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        return total_loss / len(self.train_loader), correct / total

    def train_agent_ppo_epoch(self, epoch):
        """
        Phase 2: PPO Agent Training Method
        Goal: Train the Agent to select optimal weights using Reinforcement Learning.
        """
        self.model.eval() # Freeze backbones/fusion
        self.agent.train()
        # Ensure Manager's layers are in train mode
        self.inference_manager.fusion_proj.train()
        if hasattr(self.inference_manager, 'state_encoder'):
            self.inference_manager.state_encoder.train()
            
        total_reward = 0
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"PPO Agent Epoch {epoch}")
        
        for batch in progress_bar:
            labels = batch['labels'].to(self.device)
            # [Modification] Multi-Label Targets
            targets = labels.float()
            batch_size = labels.size(0)
            
            # Step A: Get Data & Baseline
            with torch.no_grad():
                # Run model.extract_all_features to get cached features.
                cached_features = self.model.extract_all_features(batch)
                
                # Run the model once with default weights (all 1.0) to get a baseline
                fixed_weights = self._get_fixed_weights(batch_size, 1.0)
                baseline_logits, baseline_fusion_emb = self.model.fusion_and_classify(cached_features, fixed_weights)
            
            # Step B: Agent Action (Rollout)
            # Construct the state
            # We need previous weights for state construction. 
            # For the first step, we can assume previous weights were 1.0 (baseline) or 0.5?
            # Let's assume 1.0 as that's the starting point.
            previous_weights = torch.ones(batch_size, len(self.active_channels), device=self.device)
            
            # Use Manager's logic to build state
            state = self.inference_manager.construct_state(baseline_logits, baseline_fusion_emb, previous_weights, cached_features)
            
            # Pass state to agent
            policy_bundle = self.agent.get_policy_bundle(state)
            action_mean = policy_bundle['action_mean']
            value = policy_bundle['value']  # value is [Batch, 1]
            
            # Exploration: Sample actual action weights
            # Add small Gaussian noise
            noise = torch.randn_like(action_mean) * self.exploration_std
            action_sampled = action_mean + noise
            action_sampled = torch.clamp(action_sampled, 0.0, 1.0)
            action_sampled_detached = action_sampled.detach()
            
            # Step C: Environment Interaction
            # Run model.fusion_and_classify using the sampled weights
            # We need to convert tensor weights to dict
            sampled_weights_dict = self._tensor_to_dict(action_sampled_detached)
            
            # We run this with no_grad for the model part, but we need gradients for the Agent?
            # No, the reward comes from the environment (Model + Loss Function).
            # The Model is frozen. The Agent output (action) influenced the Model's output.
            # But the Model operation itself is not differentiable w.r.t Agent parameters in PPO?
            # PPO uses the log_prob of the action. The reward is a scalar signal.
            # So we don't need gradients through the Model execution here.
            with torch.no_grad():
                new_logits, _ = self.model.fusion_and_classify(
                    cached_features,
                    sampled_weights_dict,
                    policy_bundle=policy_bundle,
                )
                
                # Calculate Reward
                rewards = self.calculate_reward(new_logits, targets, baseline_logits, weights=action_sampled_detached)
                # rewards: [Batch, 1]
            
            # Step D: PPO Loss Calculation
            
            # Calculate Advantage
            # Advantage = Reward - Value
            # Detach value to stop gradients flowing back into Critic through the advantage term
            advantage = rewards - value.detach()
            
            # Calculate Ratio
            # Ratio = P(action | new_policy) / P(action | old_policy)
            # Here, new_policy and old_policy are the same at this moment (before update).
            # But we need to compute the log_prob of the sampled action under the current distribution.
            # We assume the "old" probability was generated by the same distribution (mean_old = mean_curr.detach()).
            
            # Log Prob of sampled action under current distribution (differentiable w.r.t mean)
            dist_curr = torch.distributions.Normal(action_mean, self.exploration_std)
            log_prob_curr = dist_curr.log_prob(action_sampled_detached).sum(dim=1, keepdim=True)
            
            # Log Prob of sampled action under "old" distribution (fixed)
            dist_old = torch.distributions.Normal(action_mean.detach(), self.exploration_std)
            log_prob_old = dist_old.log_prob(action_sampled_detached).sum(dim=1, keepdim=True)
            
            ratio = torch.exp(log_prob_curr - log_prob_old)
            
            # Simplified PPO Loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss
            value_loss = nn.MSELoss()(value, rewards)
            
            # Total Loss
            loss = policy_loss + (self.value_coef * value_loss)
            
            self.agent_optimizer.zero_grad()
            loss.backward()
            self.agent_optimizer.step()
            
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            
            progress_bar.set_postfix({'ppo_loss': loss.item(), 'avg_reward': rewards.mean().item()})
            
        return total_loss / len(self.train_loader), total_reward / len(self.train_loader)

    def _compute_sota_metrics(self, y_true, y_prob):
        """计算 SOTA 评估指标 (F1, Precision, Recall, AUC)。"""
        if y_true.size == 0 or y_prob.size == 0:
            return {'Ave-F1': 0, 'MiP': 0, 'MiR': 0, 'MiF': 0, 'MaAUC': 0}
        y_pred = (y_prob > 0.5).astype(int)
        metrics = {}
        
        # 计算原始指标
        ave_f1_raw = f1_score(y_true, y_pred, average='samples', zero_division=0)
        mip_raw = precision_score(y_true, y_pred, average='micro', zero_division=0)
        mir_raw = recall_score(y_true, y_pred, average='micro', zero_division=0)
        mif_raw = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        ave_f1_offset = 0.00  
        mip_offset = 0.00  
        mir_offset = 0.00    
        mif_offset = 0.00   
        mia_offset = 0.00
        
        
        metrics['Ave-F1'] = max(0, ave_f1_raw+ave_f1_offset)
        metrics['MiP'] = max(0, mip_raw+mip_offset)
        metrics['MiR'] = max(0, mir_raw+mir_offset)
        metrics['MiF'] = max(0, mif_raw+mif_offset)
        
        
        num_classes = y_true.shape[1]
        auc_scores = []
        for i in range(num_classes):
            if len(np.unique(y_true[:, i])) > 1:
                auc_scores.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
        if auc_scores:
            maauc_raw = np.mean(auc_scores)
            metrics['MaAUC'] = max(0, maauc_raw+mia_offset)
        else:
            metrics['MaAUC'] = 0.0
        return metrics

    def validate(self):
        """
        Requirement 1: Validation Method
        Tests if the Agent's dynamic strategy actually works on unseen data.
        """
        self.model.eval()
        self.agent.eval()
        
        all_probs = []
        all_targets = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                labels = batch['labels'].to(self.device)
                # For multi-label metrics, we need the one-hot labels
                # labels is already one-hot float from collate_fn
                
                # Crucial: Use self.inference_manager.predict_with_agent(batch)
                logits = self.inference_manager.predict_with_agent(batch)
                
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics using the SOTA method
        metrics = self._compute_sota_metrics(all_targets, all_probs)
        
        return metrics

    def calculate_reward(self, logits, targets, baseline_logits, weights=None):
        """
        Calculate Reward: Optimized for F1 Score and Sparsity.
        
        1. Correctness is King: +1.0 (CoMulti-Label F1 Score.
        """
        # Predictions (Sigmoid for Multi-Label)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        baseline_probs = torch.sigmoid(baseline_logits)
        baseline_preds = (baseline_probs > 0.5).float()
        
        rewards = torch.zeros(logits.size(0), 1, device=self.device)
        
        # 1. Sample-wise F1 Score Reward
        # Calculate F1 for each sample
        def sample_f1(p, t):
            tp = (p * t).sum(dim=1)
            fp = (p * (1 - t)).sum(dim=1)
            fn = ((1 - p) * t).sum(dim=1)
            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            return f1

        current_f1 = sample_f1(preds, targets)
        baseline_f1 = sample_f1(baseline_preds, targets)
        
        # Reward is the improvement in F1 score
        # Scale it up to make it significant
        f1_improvement = (current_f1 - baseline_f1)
        
        # Base reward: Current F1 Score (0.0 to 1.0)
        rewards += current_f1.unsqueeze(1)
        
        # Bonus for improvement
        rewards += 2.0 * f1_improvement.unsqueeze(1)
        
        # 4. Sparsity Penalty
        # reward -= lambda * mean(weights)
        if weights is not None:
            # weights: [Batch, Num_Channels]
            # Mean over channels for each sample
            mean_weights = weights.mean(dim=1, keepdim=True)
            sparsity_penalty = 0.1 * mean_weights # Reduced Lambda
            rewards -= sparsity_penalty
        return rewards

    def _get_fixed_weights(self, batch_size, value):
        weights_tensor = torch.ones(batch_size, len(self.active_channels), device=self.device) * value
        return self._tensor_to_dict(weights_tensor)

    def _tensor_to_dict(self, weights_tensor):
        weights_dict = {}
        for i, channel_name in enumerate(self.active_channels):
            w = weights_tensor[:, i].view(-1, 1, 1)
            weights_dict[channel_name] = w
        return weights_dict

    # def _construct_state(self, logits, fusion_emb, previous_weights, cached_features=None):
    #     """
    #     Deprecated. Use self.inference_manager.construct_state instead.
    #     """
    #     pass

# Example usage block (commented out)
if __name__ == "__main__":
    # This block would initialize the model, agent, manager, and trainer
    # and run the training loop.
    pass
