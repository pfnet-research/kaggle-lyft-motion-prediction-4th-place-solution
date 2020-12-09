# using ResNeSt-50 as an example
from resnest.torch import resnest50, resnest101, resnest200, resnest269


import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict

import sys
import os

from torch.nn import Sequential

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from lib.nn.block.linear_block import LinearBlock
from lib.nn.models.multi.multi_utils import calc_in_out_channels, calc_out_channels

model_name_dict = {
    "resnest50": resnest50,
    "resnest101": resnest101,
    "resnest200": resnest200,
    "resnest269": resnest269,
}


class ResNeStMulti(nn.Module):

    def __init__(self, cfg, num_modes=3, model_name='resnest50',
                 use_bn: bool = True,
                 hdim: int = 512,
                 pretrained=True,
                 in_channels: int = 0):
        super(ResNeStMulti, self).__init__()
        out_dim, num_preds, future_len = calc_out_channels(cfg, num_modes=num_modes)
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_preds = num_preds
        self.future_len = future_len
        self.num_modes = num_modes

        # self.conv0 = nn.Conv2d(
        #     in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.base_model = model_name_dict[model_name](pretrained=pretrained)
        # --- Replace first conv block, instead of preparing conv0 ----
        if isinstance(self.base_model.conv1, Sequential):
            conv = self.base_model.conv1[0]
            self.base_model.conv1[0] = nn.Conv2d(
                in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                padding=conv.padding, bias=True)
        else:
            conv = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                padding=conv.padding, bias=True)

        activation = F.leaky_relu
        self.do_pooling = True
        # if self.do_pooling:
        #     inch = self.base_model.last_linear.in_features
        # else:
        #     inch = None
        inch = None
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def calc_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        return x

    def forward(self, x):
        h = self.calc_features(x)

        if self.do_pooling:
            # h = torch.sum(h, dim=(-1, -2))
            h = torch.mean(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h


if __name__ == '__main__':
    # --- test instantiation ---
    from lib.utils.yaml_utils import load_yaml

    cfg = load_yaml("../../../../modeling/configs/0905_cfg.yaml")
    num_modes = 3
    model = ResNeStMulti(cfg, num_modes=num_modes)
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
