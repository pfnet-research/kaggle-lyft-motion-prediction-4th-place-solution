from typing import Tuple

from torch import nn, Tensor
import pytorch_pfn_extras as ppe

from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch


class RNNHeadMultiRegressor(nn.Module):

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix = ""

    def forward(
        self,
        image: Tensor,
        history_positions: Tensor,
        history_availabilities: Tensor,
        targets: Tensor,
        target_availabilities: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pred, confidences = self.predictor(image, history_positions, history_availabilities)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            f"{self.prefix}loss": loss.item(),
            f"{self.prefix}nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics
