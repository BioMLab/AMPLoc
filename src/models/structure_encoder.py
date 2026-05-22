import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm as GraphBatchNorm
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from torch_geometric.data import Batch

from src.features.structure_utils import (
    STRUCTURE_VIEW_INPUT_DIMS,
    STRUCTURE_VIEW_ORDER,
    StructurePreprocessor,
)


logger = logging.getLogger(__name__)


class StructureViewEncoder(nn.Module):
    def __init__(self, view_name: str, input_dim: int, hidden_dim: int, fusion_dim: int, num_classes: int, dropout: float = 0.2, heads: int = 4):
        super().__init__()
        self.view_name = view_name
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim

        self.gin_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gin = GINConv(self.gin_mlp, train_eps=True)
        self.gin_norm = GraphBatchNorm(hidden_dim)

        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            negative_slope=0.2,
        )
        self.gat_norm = GraphBatchNorm(hidden_dim)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph_batch):
        x, edge_index, batch_index = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        x = self.gin(x, edge_index)
        x = F.elu(x)
        x = self.gin_norm(x)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat(x, edge_index)
        x = F.elu(x)
        x = self.gat_norm(x)

        graph_vector = global_mean_pool(x, batch_index)
        features = self.projection(graph_vector)
        logits = self.aux_classifier(graph_vector)

        return {
            "features": features,
            "logits": logits,
        }


class StructureModalityEncoder(nn.Module):
    def __init__(self, config: Dict[str, object], num_classes: int, fusion_cfg: Dict[str, object]):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.fusion_dim = int(fusion_cfg["fusion_dim"])

        structure_cfg = config.get("intra_graph_channel", {})
        self.view_order = list(STRUCTURE_VIEW_ORDER)
        self.hidden_dim = int(structure_cfg.get("hidden_dim", 128))
        self.dropout = float(structure_cfg.get("dropout", 0.2))
        self.heads = int(structure_cfg.get("heads", 4))

        cache_dir = structure_cfg.get("cache_dir", "data/processed_structures_amploc")
        self.preprocessor = StructurePreprocessor(structure_cfg, cache_dir=cache_dir)

        self.view_encoders = nn.ModuleDict()
        for view_name in self.view_order:
            self.view_encoders[view_name] = StructureViewEncoder(
                view_name=view_name,
                input_dim=STRUCTURE_VIEW_INPUT_DIMS[view_name],
                hidden_dim=self.hidden_dim,
                fusion_dim=self.fusion_dim,
                num_classes=num_classes,
                dropout=self.dropout,
                heads=self.heads,
            )

    def _collect_samples(self, graph_batch) -> List[object]:
        if graph_batch is None:
            return []
        if hasattr(graph_batch, "to_data_list"):
            return graph_batch.to_data_list()
        if isinstance(graph_batch, list):
            return graph_batch
        return [graph_batch]

    def forward(self, graph_batch):
        samples = self._collect_samples(graph_batch)
        if not samples:
            return {
                "features": {},
                "logits": {},
                "structure_features": torch.zeros((0, len(self.view_order), self.fusion_dim), device=next(self.parameters()).device),
                "structure_feature_dict": {},
                "view_order": self.view_order,
            }

        processed_samples = []
        for sample in samples:
            sample_id = getattr(sample, "sample_id", "unknown")
            raw_sequence = getattr(sample, "raw_sequence", "")
            dbn_string = getattr(sample, "dbn_string", "")
            processed_samples.append(
                self.preprocessor.process(
                    sample_id=str(sample_id),
                    raw_sequence=str(raw_sequence),
                    dbn_string=str(dbn_string) if dbn_string is not None else "",
                )
            )

        per_view_graphs: Dict[str, List[object]] = {view_name: [] for view_name in self.view_order}
        for artifact in processed_samples:
            view_graphs = artifact["view_graphs"]
            for view_name in self.view_order:
                per_view_graphs[view_name].append(view_graphs[view_name])

        device = next(self.parameters()).device
        feature_dict: Dict[str, torch.Tensor] = {}
        logits_dict: Dict[str, torch.Tensor] = {}

        for view_name in self.view_order:
            view_batch = Batch.from_data_list(per_view_graphs[view_name]).to(device)
            out = self.view_encoders[view_name](view_batch)
            feature_dict[view_name] = out["features"]
            logits_dict[view_name] = out["logits"]

        structure_features = torch.stack([feature_dict[view_name] for view_name in self.view_order], dim=1)

        return {
            "features": feature_dict,
            "logits": logits_dict,
            "structure_features": structure_features,
            "structure_feature_dict": feature_dict,
            "view_order": self.view_order,
        }
