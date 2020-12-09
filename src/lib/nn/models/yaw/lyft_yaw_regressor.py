import torch
from torch import nn
import torch.nn.functional as F
import pytorch_pfn_extras as ppe


def _mse_cos_sin_loss(target_yaw, pred):
    # target_yaw in radians (bs,) --> target_cos_sin (bs, 2)
    # pred (bs, 2)
    target_cos_sin = torch.stack([torch.cos(target_yaw), torch.sin(target_yaw)], dim=1)
    return F.mse_loss(target_cos_sin, pred)


def _mae_cos_sin_loss(target_yaw, pred):
    # target_yaw in radians (bs,) --> target_cos_sin (bs, 2)
    # pred (bs, 2)
    target_cos_sin = torch.stack([torch.cos(target_yaw), torch.sin(target_yaw)], dim=1)
    return F.l1_loss(target_cos_sin, pred)


class LyftYawRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun: str = ""):
        super().__init__()
        self.predictor = predictor
        if lossfun == "mse":
            self.lossfun = _mse_cos_sin_loss
        elif lossfun == "mae":
            self.lossfun = _mae_cos_sin_loss
        else:
            print(f"[WARNING] Unknown lossfun {lossfun}, use mse loss...")
            self.lossfun = _mse_cos_sin_loss

        self.prefix = ""

    def forward(self, image, target_yaw, x_feat=None):
        if x_feat is None:
            pred = self.predictor(image)
        else:
            pred = self.predictor(image, x_feat)
        loss = self.lossfun(target_yaw, pred)
        metrics = {
            f"{self.prefix}loss": loss.item(),
            f"{self.prefix}mse": _mse_cos_sin_loss(target_yaw, pred).item(),
            f"{self.prefix}mae": _mae_cos_sin_loss(target_yaw, pred).item(),
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics
