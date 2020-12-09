import pretrainedmodels
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
from lib.nn.models.multi.multi_utils import calc_out_channels
from lib.nn.block.feat_module import FeatModule


class PretrainedCNNMulti(nn.Module):

    def __init__(
        self, cfg, num_modes=3, model_name='se_resnext101_32x4d',
        use_bn: bool = True,
        hdim: int = 512,
        pretrained='imagenet',
        in_channels: int = 0,
        feat_module_type: str = "none",
        feat_channels: int = -1,
    ):
        super(PretrainedCNNMulti, self).__init__()
        out_dim, num_preds, future_len = calc_out_channels(cfg, num_modes=num_modes)
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.num_preds = num_preds
        self.future_len = future_len
        self.num_modes = num_modes

        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

        # --- Replace first conv ---
        try:
            if hasattr(self.base_model, "layer0") and isinstance(self.base_model.layer0[0], nn.Conv2d):
                print("Replace self.base_model.layer0[0]...")
                # This works with SeResNeXt, but not tested with other network...
                conv = self.base_model.layer0[0]
                self.base_model.layer0[0] = nn.Conv2d(
                    in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=True)
                self.conv0 = None
            # elif hasattr(self.base_model, "conv1") and isinstance(self.base_model.conv1, nn.Conv2d):
            elif model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
                # torchvision resnet is follows...
                print("Replace base_model.conv1...")
                self.base_model.conv1 = nn.Conv2d(
                    in_channels,
                    self.base_model.conv1.out_channels,
                    kernel_size=self.base_model.conv1.kernel_size,
                    stride=self.base_model.conv1.stride,
                    padding=self.base_model.conv1.padding,
                    bias=False,
                )
                self.conv0 = None
            else:
                raise ValueError("Cannot extract first conv layer")
        except Exception as e:
            # TODO: Better to replace `base_model`'s first conv block!
            self.conv0 = nn.Conv2d(
                in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
            print(f'[WARNING ]Cannot extract first conv layer for {model_name}, use conv0 to align channel size')

        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

        self.feat_module_type = feat_module_type
        self.feat_module = FeatModule(
            feat_module_type=feat_module_type,
            channels=inch,
            feat_channels=feat_channels,
        )
        self.feat_channels = feat_channels

    def forward(self, x, x_feat=None):
        """

        Args:
            x: image feature (bs, ch, h, w)
            x_feat: (bs, ch). Additional feature (Ex. Agent type, timestamp...)

        Returns:
            h: (bs, ch)
        """
        if self.conv0 is None:
            h = x
        else:
            h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            # h = torch.sum(h, dim=(-1, -2))
            h = torch.mean(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)

        h = self.feat_module(h, x_feat)

        for layer in self.lin_layers:
            h = layer(h)
        return h


if __name__ == '__main__':
    # --- test instantiation ---
    from lib.utils.yaml_utils import load_yaml

    cfg = load_yaml("../../../../modeling/configs/0905_cfg.yaml")
    num_modes = 3
    model = PretrainedCNNMulti(cfg, num_modes=num_modes)
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
