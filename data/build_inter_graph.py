import os
import yaml
import logging
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from collections import Counter
import itertools
import re

# 导入您项目中的工具，以保持一致性
from src.features.lncmamba_utils import Tokenizer

# 设置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def get_kmer_features(sequence, k):
    """为单个序列计算k-mer频率向量"""
    all_possible_kmers = [''.join(p) for p in itertools.product('ATGC', repeat=k)]
    kmer_map = {kmer: i for i, kmer in enumerate(all_possible_kmers)}

    counts = Counter([''.join(sequence[i:i + k]) for i in range(len(sequence) - k + 1)])

    # 初始化一个全零向量
    features = np.zeros(len(all_possible_kmers))

    # 填充计数值
    total_kmers = sum(counts.values())
    if total_kmers == 0:
        return features  # 如果序列太短，返回零向量

    for kmer, count in counts.items():
        if kmer in kmer_map:
            features[kmer_map[kmer]] = count / total_kmers  # 标准化频率

    return features


def build_graph():
    logging.info("--- Starting Inter-Graph Construction ---")

    # 1. 加载配置
    with open('configs/main_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    inter_graph_cfg = config['inter_graph_channel']
    k = inter_graph_cfg['k_mer']
    m = inter_graph_cfg['num_neighbors']
    output_path = inter_graph_cfg['graph_output_path']

    logging.info(f"Configuration: k={k}, num_neighbors={m}")

    # 2. 加载和拆分数据集，以确保图只基于训练集构建
    logging.info(f"Loading dataset from: {data_cfg['csv_path']}")
    df = pd.read_csv(data_cfg['csv_path'])
    df['labels'] = df['label'].apply(lambda x: re.split(r'[,;]', x))

    # 执行与当前训练入口相同的分层抽样逻辑
    y_stratify_strings = ["_".join(sorted(labels)) for labels in df['labels']]
    label_counts = Counter(y_stratify_strings)
    safe_labels = {label for label, count in label_counts.items() if count > 1}
    stratify_indices = [i for i, label_str in enumerate(y_stratify_strings) if label_str in safe_labels]

    df_clean = df.iloc[stratify_indices].reset_index(drop=True)
    clean_labels_stratify = [y_stratify_strings[i] for i in stratify_indices]

    train_indices, _ = train_test_split(
        list(range(len(df_clean))),
        test_size=config['training']['validation_split'],
        random_state=config['training']['random_state'],
        stratify=clean_labels_stratify
    )

    df_train = df_clean.iloc[train_indices].reset_index(drop=True)
    logging.info(f"Graph will be built on {len(df_train)} training samples.")

    # 3. 为训练集中的每个序列提取k-mer特征
    logging.info(f"Extracting {k}-mer features for all training sequences...")
    node_features = []
    for seq in tqdm(df_train['sequence'], desc="Calculating k-mers"):
        node_features.append(get_kmer_features(seq, k))

    # 将特征转换为PyTorch张量
    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    logging.info(f"Node feature matrix created with shape: {x.shape}")

    # 4. 计算节点间的余弦相似度并构建边
    logging.info("Calculating cosine similarity matrix...")
    # 使用sklearn高效计算
    sim_matrix = cosine_similarity(x.numpy())
    # 将对角线（自身与自身的相似度）设置为一个很小的值，以避免被选为邻居
    np.fill_diagonal(sim_matrix, -1)

    logging.info(f"Building edges by selecting top {m} neighbors for each node...")
    source_nodes = []
    target_nodes = []

    # 找到每个节点的前m个邻居
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -m:]

    for i in range(len(df_train)):
        for j in top_k_indices[i]:
            source_nodes.append(i)
            target_nodes.append(j)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    logging.info(f"Graph created with {edge_index.shape[1]} edges.")

    # 5. 处理标签 (与当前训练入口保持一致)
    logging.info("Processing labels...")
    # 需要一个临时的Tokenizer和MultiLabelBinarizer来转换标签
    # 注意：我们仅用它来确定标签的维度和顺序，词汇表基于训练数据
    train_sequences_kmer = df_train['sequence'].apply(lambda s: ' '.join(
        [s[i:i + config['lncmamba']['k_mer']] for i in range(len(s) - config['lncmamba']['k_mer'] + 1)])).tolist()
    train_labels_text = df_train['labels'].tolist()

    temp_tokenizer = Tokenizer(train_sequences_kmer, train_labels_text)
    mlb = MultiLabelBinarizer(classes=list(temp_tokenizer.lab2id.values()))
    mlb.fit([[lab_id] for lab_id in temp_tokenizer.lab2id.values()])

    tokenized_labs_ids = temp_tokenizer.tokenize_labels(train_labels_text)
    labels_one_hot = mlb.transform(tokenized_labs_ids)
    y = torch.tensor(labels_one_hot, dtype=torch.float32)

    # 6. 创建PyTorch Geometric Data对象并保存
    graph_data = Data(x=x, edge_index=edge_index, y=y)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_data, output_path)
    logging.info(f"Inter-Graph data object successfully saved to: {output_path}")
    logging.info("--- Graph Construction Finished ---")


if __name__ == "__main__":
    build_graph()
