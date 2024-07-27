import torch.nn as nn


def act_layer(act_type: str = 'GELU'):
    if act_type == 'ReLU':
        act = nn.ReLU(inplace=True)
    elif act_type == 'SiLU':
        act = nn.SiLU(inplace=True)
    elif act_type == 'GELU':
        act = nn.GELU()
    elif act_type == 'Hardswish':
        act = nn.Hardswish(inplace=True)
    else:
        raise NotImplementedError(f'act_layer does not support {act_layer}')
    return act


def norm_layer(dim, norm_type: str = 'GN', eps=1e-5):
    if norm_type == 'BN':
        norm = nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        norm = nn.GroupNorm(1, dim, eps=eps)
    else:
        raise NotImplementedError(f'norm_type does not support {norm_type}')

    return norm
