import os
import sys
from typing import Optional

import torch
from torch import nn, Tensor
import timm
from timm.models.registry import register_model
from timm.models.helpers import build_model_with_cfg
from timm.models.resnet import BasicBlock, ResNet
from timm.models.resnet import _cfg as timm_resnet_cfg
from segmentation_models_pytorch.base.modules import SCSEModule

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir))
sys.path.append(os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from lib.nn.models.multi.multi_utils import calc_out_channels
from lib.nn.block.feat_module import FeatModule
from lib.nn.block.scea_module import SCEAModule


@register_model
def scseresnet18(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer=SCSEModule), **kwargs)
    default_cfg = timm_resnet_cfg(url='', interpolation='bicubic'),
    return build_model_with_cfg(
        ResNet, 'scseresnet18', default_cfg=default_cfg, pretrained=pretrained, **model_args
    )


@register_model
def scearesnet18(pretrained=False, **kwargs):
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer=SCEAModule), **kwargs)
    default_cfg = timm_resnet_cfg()
    return build_model_with_cfg(
        ResNet, 'scearesnet18', default_cfg=default_cfg, pretrained=pretrained, **model_args
    )


class TimmMulti(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels: int,
        num_modes=3,
        backbone: str = "resnet18",
        use_pretrained=True,
        hdim: int = 4096,
        feat_module_type: str = "none",
        feat_channels: int = -1,
    ) -> None:
        super().__init__()
        out_dim, num_preds, future_len = calc_out_channels(cfg, num_modes=num_modes)
        self.in_channels = in_channels
        self.num_modes = num_modes
        self.hdim = hdim
        self.out_dim = out_dim
        self.num_preds = num_preds
        self.future_len = future_len
        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=use_pretrained,
            num_classes=0,
            in_chans=in_channels
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.feature_dim = self.backbone.num_features
        self.dense = nn.Sequential(
            nn.Linear(self.feature_dim, hdim),
            nn.LeakyReLU(),
            nn.Linear(hdim, out_dim)
        )

        self.feat_module_type = feat_module_type
        self.feat_module = FeatModule(
            feat_module_type=feat_module_type,
            channels=self.feature_dim,
            feat_channels=feat_channels,
        )
        self.feat_channels = feat_channels

    def forward(self, x: Tensor, x_feat: Optional[Tensor] = None) -> Tensor:
        x = self.backbone.forward_features(x)
        x = self.avg_pool(x).reshape(*x.shape[:2])
        x = self.feat_module(x, x_feat)
        x = self.dense(x)
        return x
