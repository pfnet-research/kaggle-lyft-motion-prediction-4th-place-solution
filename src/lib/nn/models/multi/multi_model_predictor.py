import torch
from torch import nn, Tensor
from typing import Dict, Optional


class LyftMultiModelPredictor(nn.Module):

    target_scale: Optional[Tensor]

    def __init__(self, base_model: nn.Module, cfg: Dict, num_modes: int = 3,
                 target_scale: Optional[Tensor] = None):
        super().__init__()
        self.base_model = base_model
        self.in_channels = base_model.in_channels
        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        if target_scale is None:
            self.target_scale = None
        else:
            assert target_scale.shape == (self.future_len, 2)
            self.register_buffer("target_scale", target_scale)

    def forward(self, x, x_feat=None):
        if x_feat is None:
            h = self.base_model(x)
        else:
            h = self.base_model(x, x_feat)
        # h: (bs, num_preds(pred) + num_modes(confidence) )
        assert h.shape[1] == self.num_modes + self.num_preds

        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = h.shape
        pred, confidences = torch.split(h, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        if self.target_scale is not None:
            pred = pred * self.target_scale[None, None, :, :]
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
