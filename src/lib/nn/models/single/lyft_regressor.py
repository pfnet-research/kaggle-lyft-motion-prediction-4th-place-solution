import torch
import torch.nn.functional as F
from torch import nn
import pytorch_pfn_extras as ppe

from lib.functions.nll import pytorch_neg_multi_log_likelihood_single
from lib.functions.mse import mse_loss


class LyftRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun=mse_loss):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix = ""

    def forward(self, image, targets, target_availabilities):
        outputs = self.predictor(image).reshape(targets.shape)
        loss = self.lossfun(targets, outputs, target_availabilities)
        metrics = {
            f"{self.prefix}loss": loss.item(),
            f"{self.prefix}nll": pytorch_neg_multi_log_likelihood_single(targets, outputs, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics
