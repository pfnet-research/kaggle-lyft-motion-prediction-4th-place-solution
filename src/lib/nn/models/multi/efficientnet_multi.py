import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from lib.nn.block.linear_block import LinearBlock
from lib.nn.models.cnn_collections.efficient_net_wrapper import EfficientNetWrapper
from lib.nn.models.multi.multi_utils import calc_out_channels


class EfficientNetMulti(nn.Module):
    def __init__(
        self, cfg, num_modes=3, model_name="efficientnet-b0", use_pretrained=True, use_bn=True, hdim: int = 512,
        in_channels: int = 0
    ):
        super(EfficientNetMulti, self).__init__()
        out_dim, num_preds, future_len = calc_out_channels(cfg, num_modes=num_modes)
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_preds = num_preds
        self.future_len = future_len
        self.num_modes = num_modes

        # self.conv0 = nn.Conv2d(
        #     in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base_model = EfficientNetWrapper(
            model_name=model_name, use_pretrained=use_pretrained, in_channels=in_channels
        )
        activation = F.leaky_relu

        inch = None
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        # h = self.conv0(x)
        h = x
        h = self.base_model(h)

        for layer in self.lin_layers:
            h = layer(h)
        return h


if __name__ == "__main__":
    # --- test instantiation ---
    from lib.utils.yaml_utils import load_yaml

    cfg = load_yaml("../../../../modeling/configs/0905_cfg.yaml")
    num_modes = 3
    model = EfficientNetMulti(cfg, num_modes=num_modes)
    print(type(model))
    print(model)

    bs = 3
    in_channels = model.in_channels
    height, width = 224, 224
    device = "cuda:0"

    x = torch.rand((bs, in_channels, height, width), dtype=torch.float32).to(device)
    model.to(device)
    # pred, confidences = model(x)
    # print("pred", pred.shape, "confidences", confidences.shape)
    h = model(x)
    print("h", h.shape)
