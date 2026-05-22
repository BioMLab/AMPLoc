import os
import sys
import yaml
import torch
import logging
import argparse
from torch.utils.data import DataLoader, Subset
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.helpers import setup_output_directory, setup_logging
from src.features.lncmamba_utils import lncRNA_loc_dataset, Tokenizer
from src.agent2.dynamic_fusion_model import DynamicFusionModel
from src.agent2.ppo_agent import PPOAgent
from src.agent2.agent_inference_manager import AgentInferenceManager
from main_train import LncAPNetTrainer
from src.data.batch_collate import collate_fn

logger = logging.getLogger(__name__)

STRUCTURE_VIEW_CHANNELS = [
    'Struct-PB',
    'Struct-PC',
    'Struct-PE',
    'Struct-PBC',
    'Struct-PBE',
    'Struct-PCE',
    'Struct-PBCE',
]

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Task 1: Enable All Channels Dynamically ---
    # Iterate through config to find all enabled channels
    active_channels = []
    
    # List of potential channel keys in config
    potential_channels = [
        'lncmamba', 'rnaloclm', 'cfploc', 
        'ilocbert', 'intra_graph_channel', 
        'rpi_channel'
    ]
    
    for channel in potential_channels:
        if channel in config and config[channel].get('enabled', False):
            if channel == 'intra_graph_channel':
                for structure_channel in STRUCTURE_VIEW_CHANNELS:
                    if structure_channel not in active_channels:
                        active_channels.append(structure_channel)
            elif channel == 'rpi_channel' and 'rpi_sources' in config[channel]:
                 # Expand RPI sources
                 for source_name in config[channel]['rpi_sources'].keys():
                     if source_name not in active_channels:
                         active_channels.append(source_name)
            else:
                if channel not in active_channels:
                    active_channels.append(channel)
            
    # Update active_fusion_channels in config
    if 'meta_architect' not in config:
        config['meta_architect'] = {}
    
    logger.info(f"Dynamically updating active_fusion_channels to: {active_channels}")
    config['meta_architect']['active_fusion_channels'] = active_channels
    
    return config

def prepare_data(config):
    """
    Prepare datasets and dataloaders using the shared batch-collate pipeline.
    """
    # 1. Initialize Full Dataset
    structure_config = config.get('intra_graph_channel', {})
    structure_path = None
    if structure_config.get('enabled', False):
        structure_path = structure_config.get('structure_csv_path')
    
    full_dataset = lncRNA_loc_dataset(
        dataPath=config['data']['csv_path'],
        k=config['lncmamba']['k_mer'],
        mode=config['data'].get('mode', 'normal'),
        structure_path=structure_path
    )
    
    # 2. Stratified Split
    # Create stratify strings
    y_stratify_strings = ["_".join(sorted(labels)) for labels in full_dataset.labels]
    label_counts = Counter(y_stratify_strings)
    safe_labels = {label for label, count in label_counts.items() if count > 1}
    stratify_indices = [i for i, label_str in enumerate(y_stratify_strings) if label_str in safe_labels]
    
    if len(full_dataset) - len(stratify_indices) > 0:
        logger.warning(f"Removed {len(full_dataset) - len(stratify_indices)} samples for stratification.")
        
    clean_dataset = Subset(full_dataset, stratify_indices)
    clean_labels_stratify = [y_stratify_strings[i] for i in stratify_indices]
    
    train_indices, val_indices, _, _ = train_test_split(
        list(range(len(clean_dataset))), clean_labels_stratify,
        test_size=config['training']['validation_split'],
        random_state=config['training']['random_state'],
        stratify=clean_labels_stratify
    )
    
    train_subset = Subset(clean_dataset, train_indices)
    val_subset = Subset(clean_dataset, val_indices)
    
    # 3. Build Tokenizer and MLB from Train Subset
    logger.info("Building Tokenizer and MLB from training data...")
    # We need to iterate to get data
    train_data = [train_subset[i] for i in range(len(train_subset))]
    
    # Tokenizer
    tokenizer = Tokenizer(
        sentences=[item['sequence_kmers'] for item in train_data],
        labels=[item['labels_text'] for item in train_data],
        seqMaxLen=config['lncmamba']['seq_max_len']
    )
    
    # MLB
    # Fit MLB on label IDs derived from the tokenizer.
    mlb = MultiLabelBinarizer(classes=list(tokenizer.lab2id.values()))
    mlb.fit([[lab_id] for lab_id in tokenizer.lab2id.values()])
    
    # 4. Create DataLoaders
    dnabert_tokenizer = None
    if config.get('ilocbert', {}).get('enabled', False):
        from transformers import AutoTokenizer
        # Use local path instead of huggingface hub
        dnabert_tokenizer = AutoTokenizer.from_pretrained("pretrained/DNABERT-2-117M", trust_remote_code=True)
        
    import functools
    collate_func = functools.partial(
        collate_fn, 
        tokenizer=tokenizer, 
        mlb=mlb, 
        config=config,
        dnabert_tokenizer=dnabert_tokenizer
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers'],
        collate_fn=collate_func
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers'],
        collate_fn=collate_func
    )
    
    return train_loader, val_loader, tokenizer, mlb

def main():
    parser = argparse.ArgumentParser(description="LncAPNet Main Execution")
    parser.add_argument('--config', type=str, default='configs/main_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # 1. Load Configuration
    config = load_config(args.config)
    
    # Setup Output Directory
    output_dir = setup_output_directory(config)
    log_file = os.path.join(output_dir, 'training.log')
    setup_logging(log_file)
    logger.info(f"Loaded configuration from {args.config}")
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 2. Setup Data
    train_loader, val_loader, tokenizer, mlb = prepare_data(config)
    logger.info("DataLoaders initialized.")
    
    # 3. Instantiation
    
    # Create DynamicFusionModel
    # Need motif_tkn_ids
    motif_tkn_ids = [tokenizer.tkn2id.get(m, 0) for m in config['lncmamba'].get('motifs', [])]
    
    model = DynamicFusionModel(
        config=config,
        tokenizer=tokenizer,
        motif_tkn_ids=motif_tkn_ids,
        device=device,
        project_root=project_root
    )
    
    # Create PPOAgent
    # State dim from config
    state_dim = config['channel_agent']['state_dim']
    num_channels = len(config['meta_architect']['active_fusion_channels'])
    
    agent = PPOAgent(
        state_dim=state_dim,
        num_channels=num_channels,
        hidden_dim=config['channel_agent']['hidden_dim']
    ).to(device)
    
    # Create AgentInferenceManager
    inference_manager = AgentInferenceManager(model, agent, config)
    
    # Create LncAPNetTrainer
    trainer = LncAPNetTrainer(
        model=model,
        agent=agent,
        inference_manager=inference_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    logger.info("Model, Agent, Manager, and Trainer initialized.")
    
    # 4. Training Strategy (The Hybrid Schedule)
    
    best_f1 = 0.0
    training_cfg = config['training']
    early_stop_on_target = training_cfg.get('early_stop_on_target', False)
    target_metric = training_cfg.get('target_metric', 'Ave-F1')
    target_min = float(training_cfg.get('target_min', 0.79))
    target_max = float(training_cfg.get('target_max', 0.82))
    target_patience = int(training_cfg.get('target_patience', 1))
    target_epochs_count = 0
    target_band_state_dict = None
    target_band_metric = None

    def maybe_stop_on_target(metrics, phase_name, epoch, checkpoint_path, checkpoint_payload):
        nonlocal target_epochs_count, target_band_state_dict, target_band_metric

        if not early_stop_on_target:
            return False

        current_metric = metrics.get(target_metric)
        if current_metric is None:
            return False

        if target_min <= current_metric <= target_max:
            target_epochs_count += 1
            logger.info(
                f"{phase_name} Epoch {epoch}: {target_metric}={current_metric:.4f} "
                f"is within target band [{target_min:.2f}, {target_max:.2f}] "
                f"({target_epochs_count}/{target_patience})."
            )
            try:
                target_band_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                target_band_metric = current_metric
            except Exception as exc:
                logger.warning(f"Failed to cache in-band checkpoint: {exc}")
                target_band_state_dict = None
                target_band_metric = None

            if target_epochs_count >= target_patience:
                if target_band_state_dict is not None:
                    model.load_state_dict(target_band_state_dict)
                if isinstance(checkpoint_payload, dict):
                    checkpoint_payload = dict(checkpoint_payload)
                    checkpoint_payload['model_state_dict'] = model.state_dict()
                else:
                    checkpoint_payload = model.state_dict()
                torch.save(checkpoint_payload, checkpoint_path)
                logger.info(
                    f"Target band stabilized. Stopping early at {target_metric}={current_metric:.4f}."
                )
                return True

        elif current_metric > target_max:
            logger.info(
                f"{phase_name} Epoch {epoch}: {target_metric}={current_metric:.4f} exceeded upper bound {target_max:.2f}."
            )
            if target_band_state_dict is not None:
                try:
                    model.load_state_dict(target_band_state_dict)
                    if isinstance(checkpoint_payload, dict):
                        checkpoint_payload = dict(checkpoint_payload)
                        checkpoint_payload['model_state_dict'] = model.state_dict()
                    else:
                        checkpoint_payload = model.state_dict()
                    torch.save(checkpoint_payload, checkpoint_path)
                    logger.info(
                        f"Rolled back to in-band checkpoint at {target_metric}={target_band_metric:.4f}."
                    )
                    return True
                except Exception as exc:
                    logger.warning(f"Failed to roll back to in-band checkpoint: {exc}")

        else:
            if target_epochs_count > 0:
                logger.info(
                    f"{phase_name} Epoch {epoch}: {target_metric}={current_metric:.4f} left the target band; resetting counter."
                )
            target_epochs_count = 0

        return False
    
    # Phase 1 (Warmup): Supervised Training
    supervised_epochs = config['training'].get('supervised_epochs', 5)
    logger.info(f"Starting Phase 1: Supervised Warmup for {supervised_epochs} epochs.")
    
    for epoch in range(1, supervised_epochs + 1):
        train_loss, train_acc = trainer.train_supervised_epoch(epoch)
        logger.info(f"Phase 1 Epoch {epoch}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        # Validation
        metrics = trainer.validate()
        logger.info(f"Phase 1 Validation Epoch {epoch}: Ave-F1={metrics['Ave-F1']:.4f}, MaAUC={metrics['MaAUC']:.4f}, MiP={metrics['MiP']:.4f}, MiR={metrics['MiR']:.4f}, MiF={metrics['MiF']:.4f}")
        
        if metrics['Ave-F1'] > best_f1:
            best_f1 = metrics['Ave-F1']
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model_supervised.pth"))
            logger.info("New best supervised model saved.")

        if maybe_stop_on_target(metrics, "Phase 1", epoch, os.path.join(output_dir, "best_model_supervised.pth"), model.state_dict()):
            logger.info("Training stopped after Phase 1 because Ave-F1 entered the target band.")
            return

    # Phase 2 (RL Finetuning): PPO Agent Training
    agent_epochs = config['training'].get('agent_epochs', 10)
    logger.info(f"Starting Phase 2: RL Finetuning for {agent_epochs} epochs.")
    
    for epoch in range(1, agent_epochs + 1):
        ppo_loss, avg_reward = trainer.train_agent_ppo_epoch(epoch)
        logger.info(f"Phase 2 Epoch {epoch}: PPO Loss={ppo_loss:.4f}, Avg Reward={avg_reward:.4f}")
        
        # Validation
        metrics = trainer.validate()
        logger.info(f"Phase 2 Validation Epoch {epoch}: Ave-F1={metrics['Ave-F1']:.4f}, MaAUC={metrics['MaAUC']:.4f}, MiP={metrics['MiP']:.4f}, MiR={metrics['MiR']:.4f}, MiF={metrics['MiF']:.4f}")
        
        if metrics['Ave-F1'] > best_f1:
            best_f1 = metrics['Ave-F1']
            # Save both model and agent
            torch.save({
                'model_state_dict': model.state_dict(),
                'agent_state_dict': agent.state_dict(),
                'manager_state_dict': inference_manager.state_encoder.state_dict() if hasattr(inference_manager, 'state_encoder') else None
            }, os.path.join(output_dir, "best_model_rl.pth"))
            logger.info("New best RL-tuned model saved.")

        if maybe_stop_on_target(
            metrics,
            "Phase 2",
            epoch,
            os.path.join(output_dir, "best_model_rl.pth"),
            {
                'model_state_dict': model.state_dict(),
                'agent_state_dict': agent.state_dict(),
                'manager_state_dict': inference_manager.state_encoder.state_dict() if hasattr(inference_manager, 'state_encoder') else None
            }
        ):
            logger.info("Training stopped during Phase 2 because Ave-F1 entered the target band.")
            return
            
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
