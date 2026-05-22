#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成严格的实验数据划分 (Rigorous Data Splitting Script)

功能：
遵循严格的实验设置，生成用于深度学习模型训练和评估的固定数据划分。
本脚本不修改现有的模型训练代码，而是生成新的带有划分标记的数据文件（CSV），
确保实验的无偏性和完全可重复性。

实验设置遵循：
1. 全局固定随机种子 (SEED = 388014)。
2. 初始划分：90% 开发集 (Development Set) + 10% 严格隔离的独立测试集 (Held-out Test Set)。
   - 使用分层抽样 (Stratified Split)。
   - 独立测试集仅用于最终评估，严禁参与模型开发。
3. 开发集划分：5折交叉验证 (5-Fold CV)。
   - 在开发集 (90%数据) 上进行。
   - 每一折自动形成 80% 训练子集 (Training Split) 和 20% 验证子集 (Validation Split)。
   - 生成 'fold' 列，标记样本所属的验证折编号 (0-4)。
   
输出：
在 data/rigorous_splits/ 目录下生成：
- dataset1_held_out_test.csv : 10% 独立测试集
- dataset1_development.csv : 90% 开发集 (包含 'fold' 列)
"""

import pandas as pd
import numpy as np
import os
import re
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter

# --- 1. 全局实验设置 ---
SEED = 388014
OUTPUT_DIR = "data/rigorous_splits"
DATASETS = [
    {
        "name": "dataset1",
        "path": "data/rna_data.csv",
        "label_col": "SubCellular_Localization",
        "id_col": "Gene_ID",
        "seq_col": "Sequence"
    }
]

def parse_labels(df, label_col):
    """解析标签列，处理分号分隔或字符串列表格式"""
    def _parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            # 尝试处理类似 "['Nucleus', 'Cytoplasm']" 的格式
            if x.startswith('[') and x.endswith(']'):
                try:
                    return eval(x)
                except:
                    pass
            # 处理分号分隔 "Nucleus; Cytoplasm"
            return re.split(r';\s*|;', x)
        return []
    
    return df[label_col].apply(_parse)

def get_stratify_key(labels_list):
    """将标签列表转换为分层抽样的唯一键字符串 (排序组合)"""
    return ["_".join(sorted(labels)) for labels in labels_list]

def filter_rare_classes(df, stratify_col, min_count=2, context=""):
    """过滤掉样本数过少的类别组合"""
    counts = Counter(df[stratify_col])
    valid_classes = {cls for cls, count in counts.items() if count >= min_count}
    
    n_original = len(df)
    df_filtered = df[df[stratify_col].isin(valid_classes)].copy()
    n_filtered = len(df_filtered)
    
    if n_original - n_filtered > 0:
        print(f"[{context}] 过滤掉 {n_original - n_filtered} 个样本 (类别样本数 < {min_count})")
        print(f"[{context}] 剩余样本数: {n_filtered}")
    
    return df_filtered

def process_dataset(config):
    dataset_name = config['name']
    input_path = config['path']
    print(f"\n{'='*20} 处理数据集: {dataset_name} {'='*20}")
    print(f"读取文件: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"原始样本数: {len(df)}")
    
    # 1. 标签解析与分层键生成
    df['parsed_labels'] = parse_labels(df, config['label_col'])
    df['stratify_key'] = get_stratify_key(df['parsed_labels'])
    
    # 2. 初始过滤：为了进行 90/10 分层划分，每个类别至少需要 2 个样本 (但为了稳健性，建议更多)
    # 此处使用 min_count=2 保证 train_test_split 至少能分出 1 个到测试集
    df_clean = filter_rare_classes(df, 'stratify_key', min_count=2, context="Initial Filter")
    
    # 重置索引，保证后续操作对齐
    df_clean = df_clean.reset_index(drop=True)
    
    # 3. 划分 90% 开发集 和 10% 独立测试集
    try:
        X_dev, X_test = train_test_split(
            df_clean,
            test_size=0.10,
            stratify=df_clean['stratify_key'],
            random_state=SEED,
            shuffle=True
        )
    except ValueError as e:
        print(f"划分失败 (可能某些类别样本不足): {e}")
        # 尝试更激进的过滤
        print("尝试过滤掉样本数 < 5 的类别后重试...")
        df_clean = filter_rare_classes(df, 'stratify_key', min_count=5, context="Retry Filter")
        df_clean = df_clean.reset_index(drop=True)
        X_dev, X_test = train_test_split(
            df_clean,
            test_size=0.10,
            stratify=df_clean['stratify_key'],
            random_state=SEED,
            shuffle=True
        )

    print(f"独立测试集 (Held-out Test) 样本数: {len(X_test)} (保留用于最终无偏评估)")
    print(f"开发集 (Development Set) 样本数: {len(X_dev)} (用于5折交叉验证)")

    # 4. 保存独立测试集
    test_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_held_out_test.csv")
    X_test.to_csv(test_file, index=False)
    print(f"已保存独立测试集: {test_file}")
    
    # 5. 在开发集上进行 5 折交叉验证设置
    # 为了进行 5 折分层，我们需要确保开发集中的每个类别至少有 5 个样本
    # 重新计算开发集的 counts
    X_dev = X_dev.reset_index(drop=True) # 重置索引以便分配 fold
    counts_dev = Counter(X_dev['stratify_key'])
    
    # 检查是否有类别在开发集中少于 5 个样本
    rare_classes_dev = [cls for cls, count in counts_dev.items() if count < 5]
    if rare_classes_dev:
        print(f"警告: 开发集中有 {len(rare_classes_dev)} 个类别的样本数少于5个，无法进行严格的分层5折交叉验证。")
        print("这些样本将被标记为 fold = -1 (不参与交叉验证的验证集部分，或直接丢弃)。")
        print("为保证严格性，建议将其从交叉验证流程中移除。")
        
        # 过滤
        valid_indices = X_dev['stratify_key'].apply(lambda x: x not in rare_classes_dev)
        X_dev_filtered = X_dev[valid_indices].copy().reset_index(drop=True)
        removed_count = len(X_dev) - len(X_dev_filtered)
        print(f"已移除 {removed_count} 个样本以满足5折分层要求。有效开发集样本数: {len(X_dev_filtered)}")
        X_dev = X_dev_filtered
    
    # 初始化 Fold 列
    X_dev['fold'] = -1
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    # StratifiedKFold.split 需要 X 和 y
    # 我们使用 stratify_key 作为 y
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, X_dev['stratify_key'])):
        # val_idx 属于当前折的验证集 (20%)
        # train_idx 属于当前折的训练集 (80%)
        X_dev.loc[val_idx, 'fold'] = fold_idx
        
        print(f"  Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)} (Val Ratio: {len(val_idx)/len(X_dev):.2%})")

    # 6. 保存带有 Fold 信息的开发集
    dev_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_development.csv")
    
    # 清理中间列
    output_cols = [col for col in X_dev.columns if col not in ['parsed_labels', 'stratify_key']]
    # 确保保存 'stratify_key' 以便核查?? 不，用户需要 clean csv。
    # 最好保留 stratify_key 方便后续确认分布，但主要需要原始列 + fold。
    # 我们只保留原始列 + fold
    cols_to_save = list(pd.read_csv(input_path, nrows=0).columns) + ['fold']
    # 确保 fold 在最后
    if 'fold' in cols_to_save[:-1]: cols_to_save.remove('fold'); cols_to_save.append('fold')
        
    X_dev[cols_to_save].to_csv(dev_file, index=False)
    print(f"已保存开发集 (含 Fold 信息): {dev_file}")
    print(f"  Fold 列说明: 0-4 代表该样本在该 fold 编号下作为 20% 验证集 (Validation Split)。")
    print(f"  其余 fold 编号下，该样本作为 80% 训练集 (Training Split)。")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"全局随机种子 SEED: {SEED}")
    
    for config in DATASETS:
        process_dataset(config)
        
    print(f"\n{'='*20} 处理完成 {'='*20}")
    print(f"所有划分文件已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
