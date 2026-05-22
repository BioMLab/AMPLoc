import pandas as pd
import os


def filter_and_process_rbp_database(
        raw_data_path: str,
        processed_data_path: str,
        target_locations: dict
):
    """
    根据给定的关键词筛选和处理原始RBP数据库(TSV格式)。

    Args:
        raw_data_path (str): 原始RBP TSV/CSV文件的路径。
        processed_data_path (str): 保存处理后CSV文件的路径。
        target_locations (dict): 一个字典，键是我们的目标区域名称，
                                 值是用于搜索的关键词列表。
    """
    print("开始筛选和处理RBP数据库...")

    # 1. 加载原始数据
    try:
        # 尝试作为TSV加载，因为这是UniProt下载的默认格式
        df = pd.read_csv(raw_data_path, sep='\t')
        print(f"成功以TSV格式加载原始文件: {raw_data_path}")
    except Exception as e_tsv:
        print(f"作为TSV加载失败: {e_tsv}. 正在尝试作为CSV加载...")
        try:
            # 如果TSV加载失败，尝试作为CSV加载
            df = pd.read_csv(raw_data_path)
            print(f"成功以CSV格式加载原始文件: {raw_data_path}")
        except Exception as e_csv:
            print(f"作为CSV也加载失败: {e_csv}")
            print("请检查文件格式或路径是否正确。")
            return

    print(f"原始数据库包含 {len(df)} 个蛋白质。")

    # 重命名关键列以方便使用，并处理可能存在的名称变体
    column_mapping = {
        'Entry': 'protein_id',
        'Subcellular location [CC]': 'subcellular_location',
        'Sequence': 'sequence'
    }
    df.rename(columns=column_mapping, inplace=True)

    # 检查必需的列是否存在
    required_cols = ['protein_id', 'subcellular_location', 'sequence']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 文件中缺少必需的列。需要: {required_cols}, 实际存在: {list(df.columns)}")
        return

    # 2. 筛选和映射
    processed_data = []
    location_names = list(target_locations.keys())

    print("正在遍历蛋白质，进行筛选和映射...")
    for _, row in df.iterrows():
        location_text = row['subcellular_location']

        # 处理可能的非字符串数据 (如 NaN)
        if not isinstance(location_text, str):
            continue

        location_text_lower = location_text.lower()

        # 初始化定位向量
        location_vector = {name: 0 for name in location_names}
        is_relevant = False

        # 检查每个目标区域的关键词
        for loc_name, keywords in target_locations.items():
            if any(keyword in location_text_lower for keyword in keywords):
                location_vector[loc_name] = 1
                is_relevant = True

        # 如果蛋白质至少与一个目标区域相关，则保留
        if is_relevant:
            entry = {
                'protein_id': row['protein_id'],
                'sequence': row['sequence']
            }
            # 将定位向量作为新的列添加到条目中
            for loc_name in location_names:
                entry[f'loc_{loc_name}'] = location_vector[loc_name]
            processed_data.append(entry)

    # 3. 创建处理后的DataFrame
    if not processed_data:
        print("警告: 没有蛋白质通过筛选。请检查关键词或原始数据。")
        return

    df_processed = pd.DataFrame(processed_data)

    print(f"筛选完成。共有 {len(df_processed)} 个蛋白质被保留。")

    # 4. 保存结果
    output_dir = os.path.dirname(processed_data_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    df_processed.to_csv(processed_data_path, index=False)

    print("-" * 50)
    print("成功创建处理后的RBP数据库！")
    print(f"文件保存在: {processed_data_path}")
    print("处理后数据预览:")
    print(df_processed.head(10))  # 显示前10行以便更全面地查看
    print("-" * 50)
    # 打印每个定位的蛋白质数量统计
    print("每个定位的蛋白质数量统计 (一个蛋白质可属于多个定位):")
    for loc_name in location_names:
        count = df_processed[f'loc_{loc_name}'].sum()
        print(f"- {loc_name}: {count}")
    print("-" * 50)


if __name__ == '__main__':
    # --- 配置区域 ---
    # 根据我们最终的讨论确定的关键词映射
    LOCATION_KEYWORDS = {
        'Cytoplasm': ['cytoplasm', 'cytosol'],
        'Nucleus': ['nucleus', 'nucleolus', 'nucleoplasm'],
        'Chromatin': ['chromatin', 'chromosome'],
        'Insoluble_cytoplasm': [
            'cytoskeleton', 'stress granule', 'p-body',
            'processing body', 'actin', 'microtubule',
            'inclusion body', 'cytomatrix'
        ]
    }

    # 假设您的项目结构如下:
    # LncAPNet/
    # ├── data/
    # │   ├── raw/
    # │   │   └── rbp_database_raw.tsv  (您的原始文件)
    # │   └── processed/
    # └── src/
    #     └── data/
    #         └── process_rbp_database.py (本脚本)

    # 请根据您的实际文件路径进行调整
    # RAW_DATA_FILE_PATH = '../../data/raw/your_downloaded_file.tsv'
    # 我这里使用一个通用名称，请替换为您实际的文件名
    RAW_DATA_FILE_PATH = 'raw/RPI3.tsv'
    PROCESSED_DATA_FILE_PATH = 'processed/rbp_database_processed3.csv'

    filter_and_process_rbp_database(
        raw_data_path=RAW_DATA_FILE_PATH,
        processed_data_path=PROCESSED_DATA_FILE_PATH,
        target_locations=LOCATION_KEYWORDS
    )
