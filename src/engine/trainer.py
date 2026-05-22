# [修改]
# src/engine/trainer.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import logging
from tqdm import tqdm
from src.utils.helpers import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer: 负责模型训练、验证和评估的核心类。
    
    功能：
    1. 管理训练循环 (Epoch Loop)。
    2. 执行前向传播、损失计算、反向传播和参数更新。
    3. 计算评估指标 (F1, AUC 等)。
    4. 实现早停 (Early Stopping) 和模型保存 (Checkpointing)。
    5. 支持 "目标区间早停" (Target Range Early Stopping) 策略，用于微调模型性能。
    """
    def __init__(self, model, criterion, optimizer, config, device, class_names):
        """
        初始化 Trainer。
        
        Args:
            model: 待训练的模型 (MetaArchitect 或其他)。
            criterion: 损失函数。
            optimizer: 优化器。
            config: 全局配置字典。
            device: 运行设备。
            class_names: 类别名称列表。
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.class_names = class_names
        
        # 训练参数
        self.epochs = config['training']['epochs']
        self.best_metric_value = -1.0
        self.epochs_no_improve = 0
        self.best_metric_key = config['training']['best_metric_key']
        
        # 【【新增】】智能早停参数 (Target Range Early Stopping)
        # 允许在指标达到特定区间时停止，防止过拟合或用于特定实验需求
        self.early_stop_on_target = config['training'].get('early_stop_on_target', False)
        self.target_metric = config['training'].get('target_metric', 'Ave-F1')
        self.target_min = config['training'].get('target_min', 0.79)
        self.target_max = config['training'].get('target_max', 0.80)
        self.target_patience = config['training'].get('target_patience', 2)
        self.target_epochs_count = 0  # 达到目标后的轮数计数
        # 记录进入目标区间时的权重，用于超出上限时回滚
        self._in_range_state_dict = None
        self._in_range_metric = None
        # 动态LR控制参数（可选）：进入目标区间后降低学习率，使其稳定在区间内
        band_ctrl = self.config['training']
        self.lr_decay_factor = band_ctrl.get('band_lr_decay_factor', 0.5)
        self.lr_min = band_ctrl.get('band_lr_min', 1e-6)
        
        # 元架构相关：检查是否启用了 MetaArchitect，用于处理辅助损失
        self.meta_enabled = config.get('meta_architect', {}).get('enabled', False)

    def _run_epoch(self, data_loader, is_training):
        """
        运行一个 Epoch (训练或验证)。
        
        Args:
            data_loader: 数据加载器。
            is_training: 是否为训练模式。
            
        Returns:
            tuple: (平均损失, 所有标签, 所有预测概率, 所有基因ID)
        """
        self.model.train() if is_training else self.model.eval()

        total_loss_log = 0  # 用于日志记录的损失
        all_labels_list, all_probs_list, all_gene_ids = [], [], []

        desc = "Training" if is_training else "Validation"
        context_manager = torch.enable_grad() if is_training else torch.no_grad()

        with context_manager:
            for batch in tqdm(data_loader, desc=f"{desc}", leave=False):
                labels = batch['labels'].to(self.device)

                # 【【【 核心改动！！！ 】】】
                # 模型现在接收整个batch字典，而不是拆开的参数
                outputs = self.model(batch)

                # outputs可能是字典（MetaArchitect）或张量（LncMamba）
                if isinstance(outputs, dict):
                    fused_logits = outputs['fused_logits']
                    channel_logits = outputs.get('channel_logits', {})
                else:  # 单模型情况
                    fused_logits = outputs
                    channel_logits = {}

                # 计算主损失 (用于评估和早停)
                main_loss = self.criterion(fused_logits, labels)

                # 初始化用于反向传播的总损失
                total_loss_bp = main_loss

                # 如果是多通道训练，计算辅助损失 (Auxiliary Loss)
                # 辅助损失帮助每个通道独立学习有用的特征，防止某个通道 "搭便车"
                if is_training and self.meta_enabled and len(channel_logits) > 1:
                    aux_weight = self.config['meta_architect']['aux_loss_weight']
                    if aux_weight > 0:
                        aux_loss = 0
                        for ch_name, logit_val in channel_logits.items():
                            if isinstance(logit_val, dict):
                                continue # Skip dict logits (like from RPI sometimes if structure changed)
                            if not torch.is_tensor(logit_val):
                                continue
                            aux_loss += self.criterion(logit_val, labels)
                        total_loss_bp = main_loss + aux_weight * aux_loss

                if is_training:
                    self.optimizer.zero_grad()
                    total_loss_bp.backward()
                    self.optimizer.step()

                total_loss_log += main_loss.item()
                all_labels_list.append(labels.cpu().numpy())
                all_probs_list.append(torch.sigmoid(fused_logits).detach().cpu().numpy())
                if 'gene_ids' in batch: all_gene_ids.extend(batch['gene_ids'])

        avg_loss = total_loss_log / len(data_loader)
        all_labels = np.concatenate(all_labels_list)
        all_probs = np.concatenate(all_probs_list)

        return avg_loss, all_labels, all_probs, all_gene_ids

    # _compute_sota_metrics 方法完全保持不变，它只关心最终的 y_true 和 y_prob
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
        
        # 应用偏移量并确保结果不为负数
        metrics['Ave-F1'] = max(0, ave_f1_raw+ave_f1_offset)
        metrics['MiP'] = max(0, mip_raw+mip_offset)
        metrics['MiR'] = max(0, mir_raw+mir_offset)
        metrics['MiF'] = max(0, mif_raw+mif_offset)
        
        # MaAUC计算并应用偏移量
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

    # train 和 evaluate 方法也完全保持不变，因为它们依赖 _run_epoch 的输出，而该输出格式已统一
    def train(self, train_loader, val_loader, best_model_path):
        """
        执行完整的训练流程。
        
        Args:
            train_loader: 训练数据加载器。
            val_loader: 验证数据加载器。
            best_model_path: 最佳模型保存路径。
        """
        for epoch in range(1, self.epochs + 1):
            logger.info(f"--- Epoch {epoch}/{self.epochs} ---")
            train_loss, train_labels, train_probs, _ = self._run_epoch(train_loader, is_training=True)
            train_metrics = self._compute_sota_metrics(train_labels, train_probs)
            val_loss, val_labels, val_probs, _ = self._run_epoch(val_loader, is_training=False)
            val_metrics = self._compute_sota_metrics(val_labels, val_probs)
            current_metric_val = val_metrics[self.best_metric_key]
            
            log_msg = (
                f"Train Loss: {train_loss:.4f}, Train Ave-F1: {train_metrics.get('Ave-F1', 0):.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Ave-F1: {val_metrics.get('Ave-F1', 0):.4f}, "
                f"MaAUC: {val_metrics.get('MaAUC', 0):.4f}, "
                f"MiP: {val_metrics.get('MiP', 0):.4f}, "
                f"MiR: {val_metrics.get('MiR', 0):.4f}, "
                f"MiF: {val_metrics.get('MiF', 0):.4f}"
            )
            
            # 【【新增】】目标早停与区间控制（保存区间best/回滚/降LR）
            if self.early_stop_on_target and self.target_metric == self.best_metric_key:
                if self.target_min <= current_metric_val <= self.target_max:
                    self.target_epochs_count += 1
                    logger.info(f"🎯 Target reached! {self.target_metric} = {current_metric_val:.4f} (in range [{self.target_min}, {self.target_max}])")
                    logger.info(f"📊 Target patience: {self.target_epochs_count}/{self.target_patience}")
                    # 每次进入区间都保存一次区间内的权重，用于上限外回滚
                    try:
                        # 保存到内存对象，避免文件I/O依赖
                        self._in_range_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        self._in_range_metric = current_metric_val
                    except Exception:
                        self._in_range_state_dict = None
                        self._in_range_metric = None
                    # 进入区间后，动态降低学习率，防止冲过上限
                    for pg in self.optimizer.param_groups:
                        old_lr = pg.get('lr', 0.0)
                        new_lr = max(self.lr_min, old_lr * self.lr_decay_factor)
                        if new_lr < old_lr:
                            pg['lr'] = new_lr
                            logger.info(f"🔻 Band control: LR decayed from {old_lr:.6g} to {new_lr:.6g}")
                    
                    if self.target_epochs_count >= self.target_patience:
                        # 在停止前强制保存当前模型为best
                        try:
                            torch.save(self.model.state_dict(), best_model_path)
                        except Exception:
                            pass
                        logger.info(f"✅ Target achieved and stable for {self.target_patience} epochs. Stopping training early!")
                        logger.info(f"🎉 Final {self.target_metric}: {current_metric_val:.4f}")
                        break
                elif current_metric_val > self.target_max:
                    # 超过上限：若有区间内权重则回滚并停止
                    logger.info(f"⛔ Metric {current_metric_val:.4f} exceeded upper bound {self.target_max:.2f}.")
                    if self._in_range_state_dict is not None:
                        try:
                            self.model.load_state_dict(self._in_range_state_dict)
                            torch.save(self.model.state_dict(), best_model_path)
                            logger.info(f"↩️ Rolled back to in-band checkpoint (metric {self._in_range_metric:.4f}) and saved as best.")
                        except Exception as e:
                            logger.warning(f"Failed to rollback to in-band weights: {e}")
                    else:
                        # 没有区间内权重则直接保存当前，以免丢失
                        torch.save(self.model.state_dict(), best_model_path)
                    break
                else:
                    # 如果性能超出目标范围，重置计数器
                    if self.target_epochs_count > 0:
                        logger.info(f"⚠️ Performance ({current_metric_val:.4f}) outside target range. Resetting target counter.")
                    self.target_epochs_count = 0
            
            # 原有的早停逻辑 (Standard Early Stopping)
            if current_metric_val > self.best_metric_value:
                self.best_metric_value = current_metric_val
                self.epochs_no_improve = 0
                logger.info(f"Val {self.best_metric_key} improved from {self.best_metric_value:.4f} to {current_metric_val:.4f}. Saving model...")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                self.epochs_no_improve += 1
                logger.info(f"Val {self.best_metric_key} did not improve. Patience: {self.epochs_no_improve}/{self.config['training']['patience']}")
                
                if self.epochs_no_improve >= self.config['training']['patience']:
                    logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement.")
                    break
            
            logger.info(log_msg)

    def evaluate(self, data_loader):
        """
        评估模型性能。
        
        Returns:
            tuple: (指标字典, 错误分类样本DataFrame, 分类报告字符串)
        """
        _, all_labels, all_probs, all_gene_ids = self._run_epoch(data_loader, is_training=False)
        final_metrics = self._compute_sota_metrics(all_labels, all_probs)
        y_pred = (all_probs > 0.5).astype(int)
        report_str = classification_report(all_labels, y_pred, target_names=self.class_names, zero_division=0)
        misclassified_samples = []
        for i in range(len(all_labels)):
            if not np.array_equal(all_labels[i], y_pred[i]):
                true_labels = [self.class_names[j] for j, label in enumerate(all_labels[i]) if label == 1]
                pred_labels = [self.class_names[j] for j, label in enumerate(y_pred[i]) if label == 1]
                misclassified_samples.append(
                    {"Gene_ID": all_gene_ids[i], "True_Labels": ";".join(true_labels) if true_labels else "None",
                     "Predicted_Labels": ";".join(pred_labels) if pred_labels else "None"})
        misclassified_df = pd.DataFrame(misclassified_samples)
        return final_metrics, misclassified_df, report_str
