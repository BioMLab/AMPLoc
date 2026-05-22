

import pandas as pd
from torch.utils.data import Dataset
import re
import logging  # 【新增】

logger = logging.getLogger(__name__)  # 【新增】


class lncRNA_loc_dataset(Dataset):
    """
    LncRNA 亚细胞定位数据集类。
    
    该类负责加载和预处理 LncRNA 数据，包括序列信息、标签信息以及可选的二级结构信息。
    它继承自 torch.utils.data.Dataset，可直接用于 PyTorch 的 DataLoader。
    
    主要功能：
    1. 读取 CSV 格式的数据文件。
    2. 解析 LncRNA 序列和多标签定位信息。
    3. 将核苷酸序列转换为 k-mer 序列 (例如 3-mer: "ATCG" -> "ATC TCG")。
    4. 可选加载二级结构数据 (DBN 格式)，用于图神经网络通道。
    5. 根据模式 ('train'/'test') 过滤数据。
    """

    def __init__(self, dataPath, k, mode, structure_path=None):
        """
        初始化数据集。
        
        Args:
            dataPath (str): 主数据文件路径 (CSV 格式)。必须包含 ID、序列和标签列。
            k (int): k-mer 的大小 (例如 3, 4, 5, 6)。
            mode (str): 数据集模式，'train' (训练集) 或 'test' (测试集)。
            structure_path (str, optional): 二级结构文件路径 (CSV 格式)。
                                          如果提供，将加载 DBN 结构字符串。
        """
        # 读取主数据文件
        df = pd.read_csv(dataPath)

        # --- 标签处理 ---
        # 不同的数据集可能有不同的列名和格式，这里做兼容处理
        if 'SubCellular_Localization' in df.columns:
            # 新格式：标签以分号分隔，例如 "Nucleus; Cytoplasm"
            df['labels'] = df['SubCellular_Localization'].apply(lambda x: re.split('; |;', x))
        elif 'labels' in df.columns:
            # 旧格式：标签可能是字符串形式的列表，例如 "['Nucleus', 'Cytoplasm']"
            df['labels'] = df['labels'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        else:
            raise ValueError("CSV文件中必须包含 'SubCellular_Localization' 或 'labels' 列")

        # --- ID 列处理 ---
        if 'Gene_ID' in df.columns:
            df['id'] = df['Gene_ID']
        elif 'id' not in df.columns:
            raise ValueError("CSV文件中必须包含 'Gene_ID' 或 'id' 列")

        # --- 序列列处理 ---
        if 'Sequence' in df.columns:
            df['sequence'] = df['Sequence']
        elif 'sequence' not in df.columns:
            raise ValueError("CSV文件中必须包含 'Sequence' 或 'sequence' 列")

        # --- 数据清洗 ---
        # 移除标签为空或包含 'unknown' 的样本
        df = df[df['labels'].apply(lambda x: len(x) > 0 and 'unknown' not in x)].reset_index(drop=True)

        # --- 数据集划分过滤 ---
        # 如果 CSV 中包含 'set_type' 列，根据 mode 参数筛选训练集或测试集
        if 'set_type' in df.columns:
            if mode == 'train':
                df = df[df['set_type'] == 'train']
            elif mode == 'test':
                df = df[df['set_type'] == 'test']

        # 将数据转换为列表存储，方便索引
        self.ids = df['id'].tolist()
        self.sequences = df['sequence'].tolist()
        self.labels = df['labels'].tolist()
        self.k = k

        # --- 二级结构加载 (可选) ---
        self.structure_path = structure_path
        self.structures = {}  # 使用字典存储结构信息: {id: dbn_string}
        
        if self.structure_path:
            logger.info(f"Loading structures from: {self.structure_path}")
            try:
                struct_df = pd.read_csv(self.structure_path)
                # 创建 ID 到 DBN 字符串的映射
                # 假设结构文件包含 'id' 和 'dbn_string' 列
                self.structures = pd.Series(struct_df.dbn_string.values, index=struct_df.id).to_dict()
                logger.info(f"Successfully loaded {len(self.structures)} structures.")
            except FileNotFoundError:
                logger.error(f"结构文件未找到: {self.structure_path}。图通道将无法使用。")
                self.structures = {}
            except Exception as e:
                logger.error(f"加载或解析结构文件 {self.structure_path} 时失败: {e}")
                self.structures = {}

    def get_kmer_sentence(self, sequence):
        """
        将核苷酸序列转换为 k-mer 序列。
        
        例如: sequence="ATCG", k=3 -> "ATC TCG"
        
        Args:
            sequence (str): 原始核苷酸序列。
            
        Returns:
            str: 空格分隔的 k-mer 字符串。
        """
        return ' '.join([sequence[i:i + self.k] for i in range(len(sequence) - self.k + 1)])

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.ids)

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        
        Args:
            idx (int): 样本索引。
            
        Returns:
            dict: 包含样本信息的字典:
                - id: 序列 ID
                - raw_sequence: 原始序列字符串
                - sequence_kmers: k-mer 序列字符串
                - labels_text: 文本标签列表
                - dbn_string: 二级结构 DBN 字符串 (如果没有则为空字符串)
        """
        seq_id = self.ids[idx]
        raw_sequence = self.sequences[idx]
        
        # 生成 k-mer 序列
        sequence_kmers = self.get_kmer_sentence(raw_sequence)
        labels_text = self.labels[idx]

        # 获取 DBN 结构字符串
        dbn_string = self.structures.get(seq_id, "")

        # 这里不需要打印警告，如果某个ID没有结构，让它为空字符串即可
        # 后续的 collate_fn 会处理这种情况

        return {
            'id': seq_id,
            'raw_sequence': raw_sequence,
            'sequence_kmers': sequence_kmers,
            'labels_text': labels_text,
            'dbn_string': dbn_string
        }


class Tokenizer(object):
    """
    分词器 (Tokenizer)。
    
    负责将文本形式的 k-mer 序列和标签转换为模型可接受的数字 ID。
    
    主要功能：
    1. 构建 k-mer 词汇表 (Vocabulary)。
    2. 构建标签映射表 (Label Map)。
    3. 将 k-mer 序列转换为 Token ID 序列，并进行 Padding (填充) 和 Truncation (截断)。
    4. 生成 Attention Mask。
    5. 将文本标签转换为 Label ID。
    """
    
    def __init__(self, sentences, labels, seqMaxLen):
        """
        初始化 Tokenizer。
        
        Args:
            sentences (list): 所有的 k-mer 序列列表 (用于构建词汇表)。
            labels (list): 所有的标签列表 (用于构建标签映射)。
            seqMaxLen (int): 序列的最大长度 (用于 Padding/Truncation)。
        """
        self.sentences = sentences
        self.labels = labels
        self.seqMaxLen = seqMaxLen
        
        # 构建词汇表
        self.tkn2id, self.id2tkn, self.tknNum = self._tokenProcess(self.sentences)
        
        # 构建标签映射
        self.lab2id, self.id2lab, self.labNum = self._labelProcess(self.labels)

    def _tokenProcess(self, sentences):
        """
        构建 k-mer 词汇表。
        
        Args:
            sentences (list): k-mer 序列列表。
            
        Returns:
            tuple: (token到id的映射, id到token的映射, 词汇表大小)
        """
        # 初始化词汇表，包含特殊 Token
        # <PAD>: 填充符，ID=0
        # <UNK>: 未知符，ID=1
        tkn2id = {'<PAD>': 0, '<UNK>': 1}
        
        for s in sentences:
            # 确保输入被分割为 token 列表
            # 如果 s 是字符串 (例如 "ATC TCG")，需要 split
            tokens = s.split() if isinstance(s, str) else s
            
            for tkn in tokens:
                if tkn not in tkn2id:
                    tkn2id[tkn] = len(tkn2id)
                    
        id2tkn = {v: k for k, v in tkn2id.items()}
        tknNum = len(tkn2id)
        return tkn2id, id2tkn, tknNum

    def _labelProcess(self, labels):
        """
        构建标签映射表。
        
        Args:
            labels (list): 标签列表 (每个样本可能有多个标签)。
            
        Returns:
            tuple: (标签到id的映射, id到标签的映射, 标签数量)
        """
        lab2id = {}
        for l in labels:
            for lab in l:
                if lab not in lab2id:
                    lab2id[lab] = len(lab2id)
        id2lab = {v: k for k, v in lab2id.items()}
        labNum = len(lab2id)
        return lab2id, id2lab, labNum

    def tokenize_sentences(self, sentences):
        """
        将 k-mer 序列转换为 Token ID 序列。
        
        Args:
            sentences (list): k-mer 序列列表。
            
        Returns:
            tuple: (Token ID 列表, Attention Mask 列表)
        """
        tokenized_sentences = []
        attention_masks = []
        
        for s in sentences:
            # 确保输入被分割为 token 列表
            tokens = s.split() if isinstance(s, str) else s
            
            # 将 token 转换为 ID，未知 token 使用 <UNK>
            tokenized_s = [self.tkn2id.get(tkn, self.tkn2id['<UNK>']) for tkn in tokens]
            
            # Padding (填充) 或 Truncation (截断)
            padding_len = self.seqMaxLen - len(tokenized_s)
            
            if padding_len > 0:
                # 长度不足，进行填充
                # 序列填充 <PAD> (ID=0)
                tokenized_s = tokenized_s + [self.tkn2id['<PAD>']] * padding_len
                # Mask: 真实部分为 1，填充部分为 0
                mask = [1] * len(tokens) + [0] * padding_len
            else:
                # 长度超出，进行截断
                tokenized_s = tokenized_s[:self.seqMaxLen]
                mask = [1] * self.seqMaxLen
                
            tokenized_sentences.append(tokenized_s)
            attention_masks.append(mask)
            
        return tokenized_sentences, attention_masks

    def tokenize_labels(self, labels):
        """
        将文本标签转换为 Label ID。
        
        Args:
            labels (list): 文本标签列表。
            
        Returns:
            list: Label ID 列表。
        """
        tokenized_labels = []
        for l in labels:
            tokenized_l = [self.lab2id[lab] for lab in l]
            tokenized_labels.append(tokenized_l)
        return tokenized_labels
