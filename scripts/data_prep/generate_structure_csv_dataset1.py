import os
import pandas as pd
from tqdm import tqdm
import logging

# --- 配置区 ---
# DBN 文件所在的文件夹路径
DBN_FOLDER_PATH = "data/processed_structures_linearfold"

# 输出的 CSV 文件路径
OUTPUT_CSV_PATH = "data/structures.csv"
# --- 配置区结束 ---


# 设置简单的日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_structure_csv():
    """
    扫描 DBN 文件夹，读取每个文件的内容，并将其整理成一个 CSV 文件。
    CSV 包含两列：'id' 和 'dbn_string'。
    """
    if not os.path.isdir(DBN_FOLDER_PATH):
        logging.error(f"错误：文件夹不存在 -> {DBN_FOLDER_PATH}")
        logging.error("请确保 DBN_FOLDER_PATH 变量指向了正确的文件夹。")
        return

    logging.info(f"开始扫描文件夹: {DBN_FOLDER_PATH}")

    dbn_files = [f for f in os.listdir(DBN_FOLDER_PATH) if f.endswith('.dbn')]

    if not dbn_files:
        logging.warning(f"警告：在 {DBN_FOLDER_PATH} 中没有找到任何 .dbn 文件。")
        return

    logging.info(f"找到了 {len(dbn_files)} 个 .dbn 文件。开始处理...")

    data_records = []

    for filename in tqdm(dbn_files, desc="Processing .dbn files"):
        # 从文件名中提取 ID (例如 'ENST00000412151.1.dbn' -> 'ENST00000412151.1')
        sequence_id = os.path.splitext(filename)[0]

        file_path = os.path.join(DBN_FOLDER_PATH, filename)

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # DBN 字符串通常在文件的第三行
                if len(lines) >= 3:
                    dbn_string = lines[2].strip()
                    # 做一个简单的校验，确保是合法的 DBN 字符
                    if all(c in '.()[]{}<>' for c in dbn_string):
                        data_records.append({'id': sequence_id, 'dbn_string': dbn_string})
                    else:
                        logging.warning(f"文件 {filename} 的第三行似乎不是有效的 DBN 字符串，已跳过。")
                else:
                    logging.warning(f"文件 {filename} 内容不完整（少于3行），已跳过。")
        except Exception as e:
            logging.error(f"读取文件 {filename} 时发生错误: {e}")

    if not data_records:
        logging.error("未能从任何文件中成功提取数据。无法创建 CSV。")
        return

    # 创建 DataFrame
    df = pd.DataFrame(data_records)

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存为 CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    logging.info("-" * 50)
    logging.info(f"成功创建结构文件！")
    logging.info(f"总共处理了 {len(data_records)} 条记录。")
    logging.info(f"文件已保存至: {OUTPUT_CSV_PATH}")
    logging.info("现在您可以继续执行下一步了。")
    logging.info("-" * 50)


if __name__ == "__main__":
    create_structure_csv()
