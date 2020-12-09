import numpy as np
from torch import nn, Tensor


class TargetScaleWrapper(nn.Module):

    def __init__(
        self,
        predictor: nn.Module,
        target_scale: Tensor,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.target_scale: Tensor
        self.register_buffer("target_scale", target_scale)  # (future_len, 2)

    def forward(self, image: Tensor, history_positions: Tensor, history_availabilities: Tensor):
        """
        Args:
            image:
            history_positions: (batch_size, history_len, 2)
            history_availabilities: (batch_size, history_len)
        Returns:
            pred: (batch_size, num_modes, future_len, 2)
            confidence: (batch_size, num_modes)
        """
        assert history_positions.shape[1] <= self.target_scale.shape[0]
        history_positions = (
            history_positions / self.target_scale[np.newaxis, np.newaxis, :history_positions.shape[1], :]
        )
        pred, confidence = self.predictor(image, history_positions, history_availabilities)
        pred = pred * self.target_scale[np.newaxis, np.newaxis, :pred.shape[2], :]
        return pred, confidence
