import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):

    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w,
            kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h,
            kernel_h,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2). \
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def remove_center_sampling_locations(sampling_locations, kernel_w, kernel_h):
    idx = list(range(sampling_locations.shape[-2]))
    C = (kernel_w * kernel_h - 1) // 2
    idx = [i for i in idx if i != C and (i - C) % (C * 2 + 1) != 0]
    sampling_locations = sampling_locations[:, :, :, idx, :]
    return sampling_locations


def norm_layer(dim, norm_type: str = 'GN', eps=1e-5):
    if norm_type == 'BN':
        norm = nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        norm = nn.GroupNorm(1, dim, eps=eps)
    else:
        raise NotImplementedError(f'norm_type does not support {norm_type}')

    return norm


def act_layer(act_type: str = 'GELU'):
    if act_type == 'ReLU':
        act = nn.ReLU(inplace=True)
    elif act_type == 'SiLU':
        act = nn.SiLU(inplace=True)
    elif act_type == 'GELU':
        act = nn.GELU()
    else:
        raise NotImplementedError(f'act_layer does not support {act_layer}')
    return act


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self, x, center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(x, weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


def dcnv3_core_pytorch(x, offset, mask, kernel_h,
                       kernel_w, stride_h, stride_w, pad_h,
                       pad_w, dilation_h, dilation_w, group,
                       group_channels, offset_scale, remove_center):
    # for debug and test only,
    # need to use cuda version instead

    if remove_center and (kernel_h % 2 == 0 or kernel_w % 2 == 0 or kernel_w != kernel_h):
        raise ValueError('remove_center is only compatible with square odd kernel size.')

    x = F.pad(x, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = x.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(x.shape, x.device, kernel_h, kernel_w, dilation_h, dilation_w,
                                pad_h, pad_w, stride_h, stride_w)

    grid = _generate_dilation_grids(x.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, x.device)

    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).repeat(1, 1, 1, group * (
            kernel_h * kernel_w - remove_center)).to(x.device)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1)
    if remove_center:
        sampling_locations = remove_center_sampling_locations(sampling_locations, kernel_w=kernel_w, kernel_h=kernel_h)
    sampling_locations = sampling_locations.flatten(3, 4)
    sampling_locations = sampling_locations + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w - remove_center
    sampling_grids = 2 * sampling_locations - 1
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = x.view(N_, H_in * W_in, group * group_channels).transpose(1, 2). \
        reshape(N_ * group, group_channels, H_in, W_in)

    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).flatten(0, 1)
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()  # N, H, W, C


class DCNv3(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_type='GELU',
            norm_type='GN',
            center_feature_scale=False,
            remove_center=False
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_type
        :param norm_type
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')

        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1,
                      padding=(dw_kernel_size - 1) // 2, groups=channels),
            norm_layer(channels, norm_type),
            act_layer(act_type))

        self.offset = nn.Linear(channels, group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(channels, group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x):
        """
        intput                     (N, C, H, W)
        output                     (N, C, H, W)
        """
        N, C, H, W = x.shape

        out = self.input_proj(x.permute(0, 2, 3, 1))  # (N, C, H, W)-->  # (N, H, W, C)
        x_proj = out

        x1 = self.dw_conv(x).permute(0, 2, 3, 1)  # (N, C, H, W)-->  # (N, H, W, C)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)  # (N, H, W, C)

        out = dcnv3_core_pytorch(out, offset, mask,
                                 self.kernel_size, self.kernel_size,
                                 self.stride, self.stride,
                                 self.pad, self.pad,
                                 self.dilation, self.dilation,
                                 self.group, self.group_channels,
                                 self.offset_scale, self.remove_center)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(x1, self.center_feature_scale_proj_weight,
                                                                    self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1,
                                                                          self.channels // self.group).flatten(-2)

            out = out * (1 - center_feature_scale) + x_proj * center_feature_scale  # (N, H, W, C)

        out = self.output_proj(out).permute(0, 3, 1, 2)  # (N, H, W, C)-->(N, C, H, W)

        return out
