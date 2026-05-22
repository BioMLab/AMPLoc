import logging
import os

import torch
import torch_geometric

from src.features.graph_utils import dbn_to_graph_data

logger = logging.getLogger(__name__)


def collate_fn(batch, tokenizer, mlb, config, dnabert_tokenizer=None):
    seqs_kmers = [item['sequence_kmers'] for item in batch]
    tokenized_seqs, attention_masks = tokenizer.tokenize_sentences(seqs_kmers)

    labels_text = [item['labels_text'] for item in batch]
    tokenized_labs_ids = tokenizer.tokenize_labels(labels_text)
    labels_one_hot = mlb.transform(tokenized_labs_ids)

    raw_sequences = [item['raw_sequence'] for item in batch]
    gene_ids = [item['id'] for item in batch]

    collated_batch = {
        'input_ids': torch.tensor(tokenized_seqs, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(labels_one_hot, dtype=torch.float32),
        'gene_ids': gene_ids,
        'raw_sequences': raw_sequences,
    }

    if config.get('cfploc', {}).get('enabled', False):
        cgr_features = []
        feature_dir = config['cfploc']['feature_dir']
        for gene_id in gene_ids:
            feature_path = os.path.join(feature_dir, f"{gene_id}.pt")
            try:
                cgr_features.append(torch.load(feature_path, map_location='cpu'))
            except FileNotFoundError:
                logger.warning(f"CGR feature file not found for {gene_id}, returning zeros.")
                cgr_features.append(torch.zeros((1, config['cfploc']['in_channels'], 56, 56)))
        collated_batch['cgr_features'] = torch.cat(cgr_features, dim=0)

    if config.get('ilocbert', {}).get('enabled', False):
        if dnabert_tokenizer is None:
            raise ValueError("DNABERT tokenizer must be provided for ilocbert channel.")
        max_len = config['ilocbert']['seq_max_len']
        truncated_seqs = [seq[-max_len:] for seq in raw_sequences]
        iloc_inputs = dnabert_tokenizer(
            truncated_seqs,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        collated_batch['iloc_input_ids'] = iloc_inputs['input_ids']
        collated_batch['iloc_attention_mask'] = iloc_inputs['attention_mask']

    if config.get('intra_graph_channel', {}).get('enabled', False):
        graph_data_list = []
        valid_indices_list = []
        dbn_strings = [item['dbn_string'] for item in batch]

        for i, (seq, dbn) in enumerate(zip(raw_sequences, dbn_strings)):
            try:
                graph_data = dbn_to_graph_data(seq, dbn, sample_id=gene_ids[i])
                graph_data_list.append(graph_data)
                valid_indices_list.append(i)
            except Exception as exc:
                logger.error(
                    f"Error creating structure graph for SeqID {gene_ids[i]}, using a fallback backbone-only graph. Error: {exc}"
                )
                fallback_graph = dbn_to_graph_data(seq, "", sample_id=gene_ids[i])
                graph_data_list.append(fallback_graph)
                valid_indices_list.append(i)

        if graph_data_list:
            pyg_batch = torch_geometric.data.Batch.from_data_list(graph_data_list)
            pyg_batch.valid_indices = torch.tensor(valid_indices_list, dtype=torch.long)
            collated_batch['graph_data'] = pyg_batch
        else:
            collated_batch['graph_data'] = None

    return collated_batch