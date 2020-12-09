import segmentation_models_pytorch as smp
import torch
from torch import nn
import torch.nn.functional as F

import sys
import os

from torch.nn import Sequential
from torchvision.ops.roi_pool import roi_pool

sys.path.append(os.pardir)
sys.path.append(os.path.join(os.pardir, os.pardir))
from lib.nn.block.linear_block import LinearBlock
from lib.nn.models.multi.multi_utils import calc_out_channels


class SMPMultiAgentModel(nn.Module):
    def __init__(
        self,
        cfg,
        num_modes=3,
        in_channels: int = 0,
        hdim: int = 512,
        use_bn: bool = False,
        model_name: str = "smp_fpn",
        encoder_name: str = "resnet18",
        roi_kernel_size: float = 1.0,
    ):
        super(SMPMultiAgentModel, self).__init__()
        out_dim, num_preds, future_len = calc_out_channels(cfg, num_modes=num_modes)

        if model_name == "smp_unet":
            self.base_model = smp.Unet(encoder_name, in_channels=in_channels)
        elif model_name == "smp_fpn":
            self.base_model = smp.FPN(encoder_name, in_channels=in_channels)
        else:
            raise NotImplementedError(f"model_name {model_name} not supported in SMPMultiAgentModel")

        # HACKING, skip conv2d to get
        decoder_channels = self.base_model.segmentation_head[0].in_channels
        print("decoder_channels", decoder_channels)
        self.base_model.segmentation_head[0] = nn.Identity()

        activation = F.leaky_relu
        lin_head1 = LinearBlock(decoder_channels, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin_head2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin_head1, lin_head2)
        # self.lin_head = nn.Linear(decoder_channels, out_channels)

        self.in_channels = in_channels
        self.model_name = model_name
        self.num_preds = num_preds
        self.out_dim = out_dim
        self.future_len = future_len
        self.num_modes = num_modes
        self.roi_kernel_size = roi_kernel_size

    def forward(self, image, centroid_pixel, batch_agents):
        # (bs, ch, height, width)
        h_image = self.base_model(image)

        # (n_agents, ch)
        roi_kernel_size = self.roi_kernel_size
        if roi_kernel_size == 1.0:
            # Kernel size is 1, simply take the position of feature as agent feature.
            # TODO: how to handle agents outside of image...??
            x_pixel = torch.clamp(centroid_pixel[:, 0], 0, h_image.shape[3] - 1)
            y_pixel = torch.clamp(centroid_pixel[:, 1], 0, h_image.shape[2] - 1)
            h_points = h_image[batch_agents, :, y_pixel, x_pixel]
        else:
            # Take ROI pooling around the position to get the agent feature.
            # TODO: how to handle agents outside of image...??
            k = roi_kernel_size / 2
            x1 = torch.clamp(centroid_pixel[:, 0] - k, 0, h_image.shape[3] - 1).type(h_image.dtype)
            x2 = torch.clamp(centroid_pixel[:, 0] + k, 0, h_image.shape[3] - 1).type(h_image.dtype)
            y1 = torch.clamp(centroid_pixel[:, 1] - k, 0, h_image.shape[2] - 1).type(h_image.dtype)
            y2 = torch.clamp(centroid_pixel[:, 1] + k, 0, h_image.shape[2] - 1).type(h_image.dtype)
            boxes = torch.stack([batch_agents.type(h_image.dtype), x1, y1, x2, y2], dim=1)
            h_points = roi_pool(h_image, boxes, output_size=1)[:, :, 0, 0]

        for layer in self.lin_layers:
            h_points = layer(h_points)

        h = h_points
        # pred (n_agents)x(modes)x(time)x(2D coords)
        # confidences (n_agents)x(modes)
        n_agents, _ = h.shape
        pred, confidences = torch.split(h, self.num_preds, dim=1)
        pred = pred.view(n_agents, self.num_modes, self.future_len, 2)
        assert confidences.shape == (n_agents, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


if __name__ == '__main__':
    bs = 2
    in_channels = 4
    out_channels = 3
    image = torch.ones([bs, in_channels, 128, 128])
    model = smp.Unet("resnet18", in_channels=in_channels, classes=out_channels)  # 16
    model = smp.FPN("resnet18", in_channels=in_channels, classes=out_channels)  # 128
    decoder_channels = model.segmentation_head[0].in_channels
    print("decoder_channels", decoder_channels)
    out = model(image)
    print("out", out.shape)
