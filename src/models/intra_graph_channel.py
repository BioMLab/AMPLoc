"""Structure modality entry point for AMPLoc."""

import torch.nn as nn

from src.models.structure_encoder import StructureModalityEncoder


class IntraGraphChannel(nn.Module):
    """RNA secondary-structure modality with seven structural views."""

    def __init__(self, config, num_classes, fusion_cfg):
        super().__init__()
        self.structure_encoder = StructureModalityEncoder(config, num_classes, fusion_cfg)

    def forward(self, data):
        return self.structure_encoder(data)


