import torch
from torch import nn


class FeatModule(nn.Module):
    def __init__(
        self,
        feat_module_type: str = "none",
        channels: int = -1,
        feat_channels: int = -1,
    ):
        super(FeatModule, self).__init__()
        self.feat_module_type = feat_module_type
        if feat_module_type == "none":
            self.lin_feat = None
        elif feat_module_type == "sigmoid":
            self.lin_feat = nn.Linear(feat_channels, channels)
        elif feat_module_type == "film":
            self.lin_feat = nn.Linear(feat_channels, 2 * channels)
        else:
            raise ValueError(f"[ERROR] Unexpected value feat_module_type={feat_module_type}")

    def forward(self, h, h_feat=None):
        if self.feat_module_type == "none":
            # Do nothing
            return h
        elif self.feat_module_type == "sigmoid":
            assert h_feat is not None
            h *= torch.sigmoid(self.lin_feat(h_feat))
            return h
        elif self.feat_module_type == "film":
            assert h_feat is not None
            ch = h.shape[-1]
            h_feat = self.lin_feat(h_feat)
            gamma, beta = h_feat[:, :ch], h_feat[:, ch:]
            # gamma = torch.tanh(gamma)
            h = gamma * h + beta
            return h
        else:
            raise ValueError(f"[ERROR] Unexpected value self.feat_module_type={self.feat_module_type}")
