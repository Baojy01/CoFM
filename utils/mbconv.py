import torch
import torch.nn as nn
from .drop_path import DropPath
from .norm_act import act_layer, norm_layer


class SEModule_small(nn.Module):
    def __init__(self, channel):
        super(SEModule_small, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class SELayer(nn.Module):
    def __init__(self, channel, expansion=0.25):
        super(SELayer, self).__init__()
        mid_channel = int(channel * expansion)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, mid_channel, kernel_size=1, bias=False),
                                nn.Hardswish(inplace=True),
                                nn.Conv2d(mid_channel, channel, kernel_size=1, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        y = self.avg(x)
        y = self.fc(y)
        y = x * y.expand_as(x)
        return y


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, drop_path=0.,
                 norm_type='BN', act_type='GELU', use_se=False, bias=False):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = int(in_channels * expand_ratio)

        if stride == 1 and in_channels == out_channels:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            )

        if expand_ratio == 1:
            self.conv = nn.Sequential(norm_layer(in_channels, norm_type),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=bias),
                norm_layer(in_channels, norm_type),
                act_layer(act_type),
                SELayer(in_channels) if use_se else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            )
        else:
            self.conv = nn.Sequential(norm_layer(in_channels, norm_type),
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=bias),
                norm_layer(hidden_dim, norm_type),
                act_layer(act_type),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=bias),
                norm_layer(hidden_dim, norm_type),
                act_layer(act_type),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=bias)
                )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        out = self.downsample(x) + self.drop_path(self.conv(x))

        return out
