# preprocess_structures.py (最终的、唯一的、使用LinearFold的版本)

import os
import argparse
import logging
import signal
import pandas as pd
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

# --- 检查依赖 ---
try:
    import linearfold
except ImportError:
    print("\n错误:找不到 LinearFold 包。")
    print("请在您的Conda环境中运行: pip install linearfold-unofficial\n")
    exit(1)

# --- 配置日志记录 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_linearfold.log"), # 使用新日志名
        logging.StreamHandler()
    ]
)

# --- 定义超时处理 ---
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, timeout_handler)

def process_single_rna_linearfold(args_tuple):
    """
    【LinearFold版本】处理单条RNA序列，仅计算二级结构。
    这个函数内部绝对不包含任何'RNA'或'fold_compound'的调用。
    """
    # 在函数内部导入所需模块
    import linearfold
    
    rna_id, sequence, output_dir, timeout_seconds, max_len = args_tuple
    
    # 1. 检查序列长度
    if len(sequence) > max_len:
        return "too_long", rna_id

    # 2. 设置超时闹钟
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(timeout_seconds)

    try:
        # 3. 【核心】调用 LinearFold 进行计算
        (dbn_structure, mfe) = linearfold.fold(sequence)
        
        # 4. 计算完成，取消闹钟
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        # 5. 保存结果 (只保存.dbn文件)
        dbn_output_path = os.path.join(output_dir, f"{rna_id}.dbn")
        with open(dbn_output_path, 'w') as f:
            f.write(f">{rna_id}\n")
            f.write(f"{sequence}\n")
            f.write(f"{dbn_structure}\n")

        return "success", rna_id

    except TimeoutError:
        logging.warning(f"Skipping {rna_id}: Processing timed out after {timeout_seconds} seconds.")
        return "timeout", rna_id
    except Exception as e:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        logging.error(f"Failed to process {rna_id} with LinearFold: {e}", exc_info=False)
        return "error", rna_id

def main():
    parser = argparse.ArgumentParser(description="Preprocess RNA with LinearFold to get secondary structures.")
    parser.add_argument("--input_csv", type=str, default="rna_data.csv", help="Input CSV file. Default: 'rna_data.csv'")
    parser.add_argument("--output_dir", type=str, default="processed_structures_linearfold", help="Directory for processed files. Default: 'processed_structures_linearfold'")
    parser.add_argument("--id_col", type=str, default="Gene_ID", help="Column name for RNA ID. Default: 'Gene_ID'")
    parser.add_argument("--seq_col", type=str, default="Sequence", help="Column name for RNA sequence. Default: 'Sequence'")
    
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel processes. Default: 10.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for a single sequence. Default: 60 seconds.")
    parser.add_argument("--max_len", type=int, default=10000, help="Maximum sequence length. Default: 10000.")

    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        logging.error(f"Input CSV file not found: {args.input_csv}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Using LinearFold for preprocessing. Output will be saved to: {args.output_dir}")
    
    try:
        df = pd.read_csv(args.input_csv)
        if args.id_col not in df.columns or args.seq_col not in df.columns:
            logging.error(f"CSV must contain '{args.id_col}' and '{args.seq_col}' columns.")
            return
    except Exception as e:
        logging.error(f"Failed to read CSV {args.input_csv}: {e}")
        return

    # 过滤掉序列为空或非字符串的行
    df.dropna(subset=[args.seq_col], inplace=True)
    df = df[df[args.seq_col].apply(lambda x: isinstance(x, str) and len(x) > 0)]
    
    logging.info(f"Found {len(df)} valid RNA sequences in {args.input_csv}.")

    tasks = [
        (str(row[args.id_col]), row[args.seq_col], args.output_dir, args.timeout, args.max_len)
        for _, row in df.iterrows()
    ]

    logging.info(f"Starting preprocessing with {args.num_workers} workers...")
    results = {"success": 0, "timeout": 0, "too_long": 0, "error": 0}

    # 在macOS/Linux上，'fork'是默认值且通常更快，但'spawn'更稳定，尤其是在有复杂导入时
    # 我们可以显式设置它以增加稳定性
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置，可能会抛出RuntimeError，可以安全地忽略
        pass
    
    with Pool(processes=args.num_workers) as pool:
        for status, rna_id in tqdm(pool.imap_unordered(process_single_rna_linearfold, tasks), total=len(tasks)):
            if status in results:
                results[status] += 1

    logging.info("--- Preprocessing Finished ---")
    logging.info(f"Summary: {results['success']} succeeded, "
                 f"{results['too_long']} skipped (too long), "
                 f"{results['timeout']} skipped (timeout), "
                 f"{results['error']} failed (error).")
    logging.info(f"Check 'preprocess_linearfold.log' for details.")
    logging.info(f"Processed files are in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
