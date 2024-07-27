import math
import torch
import torch.fft as fft
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def to_odd(x: int):
    out = x + 1 if x % 2 == 0 else x
    return out


class ChannelAtt(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ChannelAtt, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = to_odd(kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        return y


class PixelAtt(nn.Module):
    def __init__(self, channels):
        super(PixelAtt, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.sig(self.conv(x))
        return out


class HighFreMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        _, _, H, W = x.size()
        out = self.max(x)
        out = self.proj(out)
        out = F.interpolate(out, [H, W], mode='bilinear')
        out = self.sig(out+x)
        return out


class CAFM(nn.Module):
    def __init__(self, channels):
        super(CAFM, self).__init__()
        self.ca = ChannelAtt(channels)
        # self.pa = PixelAtt(channels)
        self.pa = HighFreMixer(channels)

    def forward(self, x):
        out = self.ca(x)
        out = self.pa(out)
        return out


class CPE(nn.Module):
    """ Convolutional Position Encoding (CPE).
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, channel, k=3):
        super(CPE, self).__init__()
        self.proj = nn.Conv2d(channel, channel, k, 1, k // 2, groups=channel)

    def forward(self, x):
        out = self.proj(x) + x

        return out


class GroupLinear(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=8, bias=True):
        super(GroupLinear, self).__init__()
        assert in_channels % num_blocks == 0 and out_channels % num_blocks == 0
        self.num_blocks = num_blocks
        block_in_size = in_channels // num_blocks
        block_out_size = out_channels // num_blocks
        self.scale = 0.02

        self.weight = nn.Parameter(self.scale * torch.randn(num_blocks, block_in_size, block_out_size))
        if bias:
            self.bias = nn.Parameter(self.scale * torch.randn(num_blocks, block_out_size), requires_grad=True)
        else:
            self.bias = nn.Parameter(torch.zeros(num_blocks, block_out_size), requires_grad=False)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = rearrange(x, 'B (h d) H W -> B h d H W', h=self.num_blocks)
        out = torch.einsum('b k i h w, k i o->b k o h w', x, self.weight) + self.bias[:, :, None, None]
        out = rearrange(out, 'B h d H W -> B (h d) H W')
        return out


class DFTM(nn.Module):
    def __init__(self, channels, num_blocks=8):
        super(DFTM, self).__init__()
        self.cafm = CAFM(channels * 2)
        self.liners = nn.Sequential(GroupLinear(channels * 2, channels // 2, num_blocks),
                                    nn.Hardswish(inplace=True),
                                    GroupLinear(channels // 2, channels * 2, num_blocks))

        self.cpe = CPE(channels * 2)

    def forward(self, x):

        _, _, H, W = x.shape

        out = fft.rfft2(x.float(), dim=(2, 3), norm="ortho")
        out = torch.cat([out.real, out.imag], dim=1)
        
        out =  self.cafm(out) * self.liners(self.cpe(out))
        # out =  self.cafm(out) * self.liners(out)
        # out =  self.liners(out)
        
        out = torch.chunk(out, 2, dim=1)
        out = torch.stack(out, dim=-1)
        out = torch.view_as_complex(out.float())

        out = fft.irfft2(out, s=(H, W), dim=(2, 3), norm="ortho")

        return out
