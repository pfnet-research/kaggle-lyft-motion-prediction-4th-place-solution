from torch import nn
import torch.nn.functional as F

from lib.functions.residual_add import residual_add


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 bias=True, use_bn=True,
                 activation=F.relu, dropout_ratio=-1, residual=False, padding_mode='zeros'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias, padding_mode=padding_mode)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout2d(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def forward(self, x):
        if self.use_bn:
            h = self.bn(self.conv(x))
        else:
            h = self.conv(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h
