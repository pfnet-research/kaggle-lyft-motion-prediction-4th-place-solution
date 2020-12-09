from typing import Sequence, Tuple, Optional

import torch
from torch import nn, Tensor

from lib.nn.models.multi.multi_model_predictor import LyftMultiModelPredictor


# NOTE: Since there is no initial variation in the trained model,
# the variability is incorporated by converting the input by one of the dihedral group D4
D4 = [
    lambda x: x,  # (x, y)
    lambda x: x.transpose(2, 3).flip(3),  # (y, -x)
    lambda x: x.flip(2).flip(3),  # (-x, -y)
    lambda x: x.transpose(2, 3).flip(2),  # (-y, x)
    lambda x: x.flip(3),  # (x, -y)
    lambda x: x.transpose(2, 3),  # (y, x)
    lambda x: x.flip(2),  # (-x, y)
    lambda x: x.transpose(2, 3).flip(2).flip(3),  # (-y, -x)
]
D4_inv = [D4[k] for k in [0, 3, 2, 1, 4, 5, 6, 7]]


class D4Module(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: Tensor):
        x = D4[self.k](x)
        return x


class LyftMultiDeepEnsemblePredictor(nn.Module):

    def __init__(self, predictors: Sequence[LyftMultiModelPredictor], names: Sequence[str], use_D4: bool = False):
        super().__init__()

        if len(predictors) > 8:
            raise ValueError("We only support up to 8 models yet.")

        self.predictors = nn.ModuleList(predictors)
        self.names = names
        self.use_D4 = use_D4

    @torch.no_grad()
    def load_state_dict(self, to_load, strict: bool = True) -> None:
        if any([key.startswith("predictors") for key in to_load.keys()]):
            super().load_state_dict(to_load, strict=strict)
            return

        print("Loading from non ensemble snapshot")
        old_snapshot = True
        for key in to_load.keys():
            if key.startswith("base_model."):
                old_snapshot = False

        for predictor in self.predictors:
            if old_snapshot:
                predictor.base_model.load_state_dict(to_load, strict=strict)
            else:
                predictor.load_state_dict(to_load, strict=strict)

        # for k, predictor in enumerate(self.predictors):
        #     for module in predictor.modules():
        #         if isinstance(module, nn.Conv2d):
        #             module.weight[:] = D4[k](module.weight)

    def forward(self, x: Tensor, x_feat: Optional[Tensor] = None) -> Sequence[Tuple[Tensor, Tensor]]:
        ys = []
        for k, predictor in enumerate(self.predictors):
            if self.use_D4:
                x = D4[k](x)
            ys.append(predictor(x, x_feat))
        return ys

    def get_kth_predictor(self, k: int):
        if k >= len(self.predictors):
            raise ValueError
        if self.use_D4:
            return nn.Sequential(D4Module(k), self.predictors[k])
        else:
            return self.predictors[k]
