from torch import nn
import pytorch_pfn_extras as ppe

from lib.functions.nll import pytorch_neg_multi_log_likelihood_batch
from lib.nn.models.deep_ensemble.lyft_multi_deep_ensemble_predictor import LyftMultiDeepEnsemblePredictor


class LyftMultiDeepEnsembleRegressor(nn.Module):

    def __init__(self, predictor: LyftMultiDeepEnsemblePredictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun
        self.prefix = ""

    def forward(self, image, targets, target_availabilities, x_feat=None):
        if x_feat is None:
            predictions = self.predictor(image)
        else:
            predictions = self.predictor(image, x_feat)

        metrics = {}
        total_loss = 0.0
        for name, (pred, confidences) in zip(self.predictor.names, predictions):
            loss = self.lossfun(targets, pred, confidences, target_availabilities)
            total_loss += loss
            metrics[f"{name}/{self.prefix}loss"] = loss.item()
            metrics[f"{name}/{self.prefix}nll"] = pytorch_neg_multi_log_likelihood_batch(
                targets, pred, confidences, target_availabilities
            ).item()

        ppe.reporting.report(metrics, self)
        return total_loss, metrics
