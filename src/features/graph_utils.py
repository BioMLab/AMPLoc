# src/features/graph_utils.py

import torch
from torch_geometric.data import Data

from src.features.structure_utils import truncate_for_structure

# 核苷酸到one-hot编码的映射
NUCLEOTIDE_MAP = {
    'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'T': [0, 1, 0, 0],  # T和U视为相同
    'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],  # 对于未知核苷酸使用零向量
}
NODE_DIM = 4


def dbn_to_graph_data(rna_sequence: str, dbn_string: str, sample_id: str = None, max_len: int = 3000) -> Data:
    """
    将 RNA 序列和其点括号表示法 (DBN) 转换为 PyTorch Geometric 的 Data 对象。
    
    功能：
    构建 RNA 的二级结构图。
    - 节点 (Nodes): 代表 RNA 序列中的核苷酸 (A, U, C, G)。
    - 边 (Edges): 代表化学键。
        1. 磷酸二酯键 (Backbone): 连接序列中相邻的核苷酸 (i -> i+1)。
        2. 氢键 (Pairing): 连接由括号配对的碱基对 (i <-> j)。
        
    Args:
        rna_sequence (str): RNA 序列字符串 (如 "AUCG...")。
        dbn_string (str): 对应的二级结构 DBN 字符串 (如 "((..))...")。

    Returns:
        torch_geometric.data.Data: 包含节点特征 (x) 和边索引 (edge_index) 的图数据对象。
    """
    rna_sequence = truncate_for_structure(rna_sequence, max_len=max_len)
    seq_len = len(rna_sequence)
    dbn_string = (dbn_string or "")[:seq_len]

    # 1. 创建节点特征 (Node Features, x)
    # 将每个核苷酸转换为 one-hot 编码
    node_features = [NUCLEOTIDE_MAP.get(n, NUCLEOTIDE_MAP['N']) for n in rna_sequence]
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. 创建边索引 (Edge Index, edge_index)
    source_nodes = []
    target_nodes = []

    # 添加骨干连接 (backbone edges)
    for i in range(seq_len - 1):
        source_nodes.append(i)
        target_nodes.append(i + 1)

    # 添加氢键连接 (pairing edges from dbn)
    stack = []
    for i, char in enumerate(dbn_string):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                source_nodes.append(j)
                target_nodes.append(i)

    # PyG要求边是双向的，以更好地传播信息
    edge_sources = source_nodes + target_nodes
    edge_targets = target_nodes + source_nodes

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        sample_id=str(sample_id) if sample_id is not None else "",
        raw_sequence=rna_sequence,
        dbn_string=dbn_string,
        original_sequence_length=seq_len,
        structure_sequence_length=seq_len,
    )

