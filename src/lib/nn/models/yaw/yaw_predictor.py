import torch
from torch import nn, Tensor
from typing import Dict, Optional


class LyftYawPredictor(nn.Module):

    target_scale: Optional[Tensor]

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.in_channels = base_model.in_channels
        # X, Y coords for the future positions (output shape: Bx50x2)

    def forward(self, x, x_feat=None):
        if x_feat is None:
            h = self.base_model(x)
        else:
            h = self.base_model(x, x_feat)
        # h: (bs, num_preds(pred) + num_modes(confidence) --> Only return 2)
        return h[:, :2]
