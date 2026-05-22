
import sys
import os
import yaml
import logging
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# Adjust path to include src
sys.path.append(os.getcwd())

from src.features.lncmamba_utils import lncRNA_loc_dataset

# Setup minimal logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def inspect_split(config_path, dataset_name):
    print(f"\n--- Inspecting {dataset_name} ({config_path}) ---")
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    csv_path = config['data']['csv_path']
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return

    print(f"Data Source: {csv_path}")
    
    # Load raw dataframe to get total count
    df = pd.read_csv(csv_path)
    total_raw = len(df)
    print(f"Total raw samples: {total_raw}")

    # Use dataset class logic
    try:
        full_dataset = lncRNA_loc_dataset(
            dataPath=csv_path,
            k=config.get('lncmamba', {}).get('k_mer', 3), # Default k=3 if missing
            mode='train' # Assume training mode for split inspection
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Replicate the current training entrypoint's filtering logic.
    # Note: lncRNA_loc_dataset itself does not filter, it just loads.
    # The filtering happens before the split based on stratify labels.

    labels_list = []
    # Re-implement label extraction logic from dataset if not directly accessible or to be safe
    # But usually dataset stores labels in .labels
    if hasattr(full_dataset, 'labels'):
        labels_list = full_dataset.labels
    else:
        # Fallback if implementation changed - unlikely based on read_file
        print("Could not access dataset labels.")
        return

    y_stratify_strings = ["_".join(sorted(labels)) for labels in labels_list]
    label_counts = Counter(y_stratify_strings)
    
    # Logic mirrored from the current train/validation split pipeline.
    safe_labels = {label for label, count in label_counts.items() if count > 1}
    stratify_indices = [i for i, label_str in enumerate(y_stratify_strings) if label_str in safe_labels]
    
    removed_count = total_raw - len(stratify_indices)
    print(f"Samples removed (single-instance label combos): {removed_count}")
    print(f"Effective samples for splitting: {len(stratify_indices)}")

    if len(stratify_indices) == 0:
        print("No samples left after filtering!")
        return

    clean_labels_stratify = [y_stratify_strings[i] for i in stratify_indices]
    
    val_split = config['training'].get('validation_split', 0.2)
    random_state = config['training'].get('random_state', 42)

    train_indices, val_indices, _, _ = train_test_split(
        list(range(len(stratify_indices))), 
        clean_labels_stratify,
        test_size=val_split,
        random_state=random_state,
        stratify=clean_labels_stratify
    )
    
    print(f"Split Ratio: {1-val_split:.0%}/{val_split:.0%} (Train/Val)")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Check for independent test set mention in config or related configs
    # Heuristic: Check if there's a related RPI test config or similar
    # (This part is manual interpretation usually, but we can check adjacent files)
    
    return len(train_indices), len(val_indices)

if __name__ == "__main__":
    # Dataset 1
    inspect_split("configs/main_config.yaml", "Dataset 1")
