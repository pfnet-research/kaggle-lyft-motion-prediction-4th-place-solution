import torch
from torch import nn
import pytorch_pfn_extras as ppe

from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch
from lib.functions.mse import mse_loss_multi


class LyftMultiRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix = ""

    def forward(self, image, targets, target_availabilities, x_feat=None):
        if x_feat is None:
            pred, confidences = self.predictor(image)
        else:
            pred, confidences = self.predictor(image, x_feat)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            f"{self.prefix}loss": loss.item(),
            f"{self.prefix}nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics
