#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集数据分割脚本
将lncRNA_Validation.csv和validation_rpi_scores.csv按照1:1比例分割为训练集和测试集
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def load_validation_data():
    """加载验证集数据"""
    print("正在加载验证集数据...")
    
    # 数据文件路径
    validation_dir = Path("temp_test_files/Validation set")
    lncrna_file = validation_dir / "lncRNA_Validation.csv"
    rpi_file = validation_dir / "validation_rpi_scores.csv"
    
    # 检查文件是否存在
    if not lncrna_file.exists():
        raise FileNotFoundError(f"找不到文件: {lncrna_file}")
    if not rpi_file.exists():
        raise FileNotFoundError(f"找不到文件: {rpi_file}")
    
    # 加载数据
    lncrna_df = pd.read_csv(lncrna_file)
    rpi_df = pd.read_csv(rpi_file)
    
    print(f"lncRNA数据形状: {lncrna_df.shape}")
    print(f"RPI数据形状: {rpi_df.shape}")
    
    return lncrna_df, rpi_df

def align_data(lncrna_df, rpi_df):
    """对齐lncRNA数据和RPI数据"""
    print("正在对齐数据...")
    
    # 获取基因ID
    lncrna_ids = set(lncrna_df['Gene_ID'].values)
    rpi_ids = set(rpi_df.iloc[:, 0].values)  # 第一列是基因ID
    
    print(f"lncRNA数据中的基因ID数量: {len(lncrna_ids)}")
    print(f"RPI数据中的基因ID数量: {len(rpi_ids)}")
    
    # 找到交集
    common_ids = lncrna_ids.intersection(rpi_ids)
    print(f"共同基因ID数量: {len(common_ids)}")
    
    # 过滤数据，只保留共同的基因ID
    lncrna_filtered = lncrna_df[lncrna_df['Gene_ID'].isin(common_ids)].reset_index(drop=True)
    rpi_filtered = rpi_df[rpi_df.iloc[:, 0].isin(common_ids)].reset_index(drop=True)
    
    # 确保两个数据集的基因ID顺序一致
    lncrna_sorted = lncrna_filtered.sort_values('Gene_ID').reset_index(drop=True)
    rpi_sorted = rpi_filtered.sort_values(rpi_df.columns[0]).reset_index(drop=True)
    
    print(f"对齐后lncRNA数据形状: {lncrna_sorted.shape}")
    print(f"对齐后RPI数据形状: {rpi_sorted.shape}")
    
    return lncrna_sorted, rpi_sorted

def split_data(lncrna_df, rpi_df, test_size=0.5, random_state=42):
    """按照1:1比例分割数据"""
    print(f"正在按照1:1比例分割数据 (测试集比例: {test_size})...")
    
    # 获取基因ID列表
    gene_ids = lncrna_df['Gene_ID'].values
    
    # 分割索引
    train_indices, test_indices = train_test_split(
        range(len(gene_ids)), 
        test_size=test_size, 
        random_state=random_state,
        stratify=lncrna_df['SubCellular_Localization']  # 按标签分层
    )
    
    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")
    
    # 分割lncRNA数据
    lncrna_train = lncrna_df.iloc[train_indices].reset_index(drop=True)
    lncrna_test = lncrna_df.iloc[test_indices].reset_index(drop=True)
    
    # 分割RPI数据
    rpi_train = rpi_df.iloc[train_indices].reset_index(drop=True)
    rpi_test = rpi_df.iloc[test_indices].reset_index(drop=True)
    
    return lncrna_train, lncrna_test, rpi_train, rpi_test

def save_split_data(lncrna_train, lncrna_test, rpi_train, rpi_test):
    """保存分割后的数据"""
    print("正在保存分割后的数据...")
    
    # 创建输出目录
    output_dir = Path("temp_test_files/split_validation")
    output_dir.mkdir(exist_ok=True)
    
    # 保存lncRNA数据
    lncrna_train.to_csv(output_dir / "lncRNA_Train.csv", index=False)
    lncrna_test.to_csv(output_dir / "lncRNA_Test.csv", index=False)
    
    # 保存RPI数据
    rpi_train.to_csv(output_dir / "rpi_scores_Train.csv", index=False)
    rpi_test.to_csv(output_dir / "rpi_scores_Test.csv", index=False)
    
    print(f"数据已保存到: {output_dir}")
    print("文件列表:")
    print("- lncRNA_Train.csv (训练集lncRNA数据)")
    print("- lncRNA_Test.csv (测试集lncRNA数据)")
    print("- rpi_scores_Train.csv (训练集RPI数据)")
    print("- rpi_scores_Test.csv (测试集RPI数据)")

def analyze_split_data(lncrna_train, lncrna_test, rpi_train, rpi_test):
    """分析分割后的数据"""
    print("\n=== 数据分割分析 ===")
    
    # 分析lncRNA数据
    print(f"训练集lncRNA数据: {lncrna_train.shape}")
    print(f"测试集lncRNA数据: {lncrna_test.shape}")
    
    # 分析标签分布
    print("\n训练集标签分布:")
    train_label_counts = lncrna_train['SubCellular_Localization'].value_counts()
    print(train_label_counts)
    
    print("\n测试集标签分布:")
    test_label_counts = lncrna_test['SubCellular_Localization'].value_counts()
    print(test_label_counts)
    
    # 分析RPI数据
    print(f"\n训练集RPI数据: {rpi_train.shape}")
    print(f"测试集RPI数据: {rpi_test.shape}")
    
    # 检查数据完整性
    print("\n数据完整性检查:")
    print(f"训练集基因ID匹配: {len(set(lncrna_train['Gene_ID']).intersection(set(rpi_train.iloc[:, 0])))}")
    print(f"测试集基因ID匹配: {len(set(lncrna_test['Gene_ID']).intersection(set(rpi_test.iloc[:, 0])))}")

def main():
    """主函数"""
    print("=== 验证集数据分割脚本 ===")
    print("将lncRNA_Validation.csv和validation_rpi_scores.csv按照1:1比例分割")
    print()
    
    try:
        # 1. 加载数据
        lncrna_df, rpi_df = load_validation_data()
        
        # 2. 对齐数据
        lncrna_aligned, rpi_aligned = align_data(lncrna_df, rpi_df)
        
        # 3. 分割数据
        lncrna_train, lncrna_test, rpi_train, rpi_test = split_data(
            lncrna_aligned, rpi_aligned, test_size=0.5, random_state=42
        )
        
        # 4. 保存数据
        save_split_data(lncrna_train, lncrna_test, rpi_train, rpi_test)
        
        # 5. 分析数据
        analyze_split_data(lncrna_train, lncrna_test, rpi_train, rpi_test)
        
        print("\n=== 数据分割完成 ===")
        print("现在可以使用分割后的数据进行RPI模态训练了！")
        
    except Exception as e:
        print(f"错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
