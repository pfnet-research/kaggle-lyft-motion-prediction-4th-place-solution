from typing import Tuple, Dict

import torch
from torch import nn, Tensor
import pytorch_pfn_extras as ppe

from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch


class LyftMultiAgentRegressor(nn.Module):
    """Multi agent, multi mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix = ""

    def forward(
        self,
        image: Tensor,
        centroid_pixel: Tensor,
        batch_agents: Tensor,
        targets: Tensor,
        target_availabilities: Tensor
    ) -> Tuple[Tensor, Dict]:
        """

        Args:
            image: (batch_size=n_frames, ch, height, width)
            centroid_pixel: (n_agents, coords=2)
            batch_agents: (n_agents,)
            targets: (n_agents,)
            target_availabilities: (n_agents, future_len=50)

        Returns:
            loss:
            metrics:
        """
        # pred (n_agents)x(modes)x(time)x(2D coords)
        # confidences (n_agents)x(modes)
        pred, confidences = self.predictor(image, centroid_pixel, batch_agents)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            f"{self.prefix}loss": loss.item(),
            f"{self.prefix}nll": pytorch_neg_multi_log_likelihood_batch(
                targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics
