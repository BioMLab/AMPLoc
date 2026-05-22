import logging
import torch
import os
import yaml
from datetime import datetime
import pandas as pd  


def setup_output_directory(config):
    """
    设置输出目录。
    
    功能：
    1. 根据当前时间戳创建一个新的输出目录，避免覆盖旧的实验结果。
    2. 将本次运行的配置 (config) 保存到该目录下，确保实验的可复现性。
    
    Args:
        config (dict): 配置字典。
        
    Returns:
        str: 创建的输出目录路径。
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = config['output']['dir']
    output_dir = os.path.join(base_dir, run_timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save the config file for this run for reproducibility
    with open(os.path.join(output_dir, 'config_run.yaml'), 'w') as f:
        yaml.dump(config, f)

    return output_dir


def setup_logging(log_path):
    """
    配置日志系统。
    
    功能：
    配置 logging 模块，使其同时将日志输出到控制台和指定的文件中。
    
    Args:
        log_path (str): 日志文件的保存路径。
    """
    # Remove all existing handlers to prevent duplicate logs in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    # Suppress overly verbose logs from libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("gensim").setLevel(logging.WARNING)  # 抑制gensim的日志


def save_checkpoint(model, optimizer, epoch, best_metric, filepath):
    """
    保存模型检查点 (Checkpoint)。
    
    功能：
    保存模型当前的权重、优化器状态、当前轮数和最佳指标值。
    这允许在训练中断后恢复训练，或在推理时加载最佳模型。
    
    Args:
        model: 模型对象。
        optimizer: 优化器对象。
        epoch (int): 当前轮数。
        best_metric (float): 当前最佳指标值。
        filepath (str): 保存路径。
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(state, filepath)
    # Logging is handled by the caller (Trainer) which has more context.


def get_class_weights(df, label_col, label_map):
    """
    计算类别权重 (Class Weights)。
    
    功能：
    为多标签分类任务计算 BCEWithLogitsLoss 的 pos_weight 参数。
    用于解决数据不平衡问题。
    计算公式: weight = (num_negative_samples / num_positive_samples)
    
    Args:
        df (pd.DataFrame): 包含标签的数据框。
        label_col (str): 标签列的名称。
        label_map (dict): 标签到索引的映射字典。
        
    Returns:
        torch.Tensor: 每个类别的权重张量。
    """
    all_labels = []
    df[label_col].apply(lambda x: all_labels.extend(eval(x)))

    label_counts = pd.Series(all_labels).value_counts()

    # 确保所有在label_map中的类别都被考虑
    weights = torch.ones(len(label_map))
    for label, i in label_map.items():
        if label in label_counts:
            n_pos = label_counts[label]
            n_neg = len(df) - n_pos
            if n_pos > 0:
                weights[i] = n_neg / n_pos
    return weights