from torch import Tensor
import torch.nn.functional as F


# --- Single ---
def mse_loss(gt: Tensor, pred: Tensor, avails: Tensor) -> Tensor:
    loss = F.mse_loss(gt[avails > 0.], pred[avails > 0.])
    return loss


# --- Multi ---
def mse_loss_multi(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
    loss = F.mse_loss(gt[avails > 0.], pred[avails > 0.])
    return loss
