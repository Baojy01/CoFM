import torch
import torch.nn as nn
from utils import DropPath, MBConv, DFTM, DCNv3


class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStem, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.GELU(),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)        
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x_list = torch.split(x, self.channels, dim=1)
        out_list = [conv(y) for conv, y in zip(self.proj, x_list)]
        out = torch.cat(out_list, dim=1)
        return out


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class IRFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, kernel_size=1),
                                   nn.GELU(),
                                   nn.BatchNorm2d(hidden_features))
        
        self.proj = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act_bn = nn.Sequential(nn.GELU(), nn.BatchNorm2d(hidden_features))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, kernel_size=1),
                                   nn.BatchNorm2d(out_features))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.act_bn(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class MS_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MS_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
                                nn.GELU(),
                                nn.BatchNorm2d(hidden_features))

        self.dwconv = MultiScaleDWConv(hidden_features)
        
        self.norm = nn.BatchNorm2d(hidden_features)
        self.act = nn.GELU()
        
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        
        x = self.dwconv(x) + x
        
        x = self.norm(self.act(x))
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class DCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4, drop=0., drop_path=0., layer_scale=None, post_norm=False):
        super().__init__()

        G = dim // 16
        
        hidden_dim = int(dim * mlp_ratio)
        
        self.dcn = DCNv3(dim, kernel_size, stride=1, pad=1, group=G, norm_type='GN')
        self.norm1 = nn.GroupNorm(1, dim)

        self.norm2 = nn.GroupNorm(1, dim)

        self.ms_ffn = MS_FFN(dim, hidden_features=hidden_dim, drop=drop)

        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.post_norm = post_norm

    def forward(self, x):

        if self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.ms_ffn(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.ms_ffn(self.norm2(x)))
        else:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.norm2(self.ms_ffn(x)))
            else:
                x = x + self.drop_path(self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.ms_ffn(self.norm2(x)))

        return x


class FreBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, drop_path=0.,
                 layer_scale_init_value=1e-5):
        super(FreBlock, self).__init__()

        self.mbconv = MBConv(in_channels, out_channels, stride, expand_ratio, drop_path=drop_path)

        self.norm = nn.GroupNorm(1, out_channels)
        self.filter = DFTM(out_channels, num_blocks=8)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value is not None and type(layer_scale_init_value) in [int, float]:
            self.ls = True
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(out_channels, 1, 1))
        else:
            self.ls = False

    def forward(self, x):
        # input shape [B C H W]

        x = self.mbconv(x)

        if self.ls:
            x = x + self.drop_path(self.gamma * self.filter(self.norm(x)))
        else:
            x = x + self.drop_path(self.filter(self.norm(x)))

        return x


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class DualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=4, mlp_ratio=4, drop=0., drop_path=0.,
                 layer_scale_init_value=1e-5, post_norm=True, use_dcn=True):
        super(DualBlock, self).__init__()

        self.block1 = FreBlock(in_channels, out_channels, stride, expand_ratio, drop_path, layer_scale_init_value)
        if use_dcn:
            self.block2 = DCNBlock(out_channels, kernel_size, mlp_ratio, drop, drop_path, layer_scale_init_value, post_norm)
        else:
            self.block2 = FreBlock(out_channels, out_channels, 1, expand_ratio, drop_path, layer_scale_init_value)

    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)

        return out


def make_layers(in_channels, out_channels, layers, expand_ratio, drop, dpr_list, stride=1,
                layer_scale_init_value=1e-5, block_type='C', use_dcn=True):
    assert block_type in ['C', 'T']
    blocks = []

    if block_type == 'C':
        blocks.append(
            MBConv(in_channels, out_channels, stride=stride, expand_ratio=expand_ratio, drop_path=dpr_list[0]))
        for block_idx in range(1, layers):
            blocks.append(
                MBConv(out_channels, out_channels, stride=1, expand_ratio=expand_ratio, drop_path=dpr_list[block_idx]))
            
    if block_type == 'T':
        blocks.append(DualBlock(in_channels, out_channels, stride=stride, expand_ratio=expand_ratio, drop=drop,
                                drop_path=dpr_list[0], layer_scale_init_value=layer_scale_init_value, use_dcn=use_dcn))

        for block_idx in range(1, layers):
            blocks.append(
                DualBlock(out_channels, out_channels, stride=1, expand_ratio=expand_ratio, drop=drop,
                          drop_path=dpr_list[block_idx], layer_scale_init_value=layer_scale_init_value, use_dcn=use_dcn))
            
    return nn.Sequential(*blocks)


def split_list(init_list: list, split: list):
    count = 0
    out_list = []
    for i in split:
        sub_list = init_list[count:i + count]
        out_list.append(sub_list)
        count += i

    return out_list


class CoFM(nn.Module):
    arch_settings = {
        **dict.fromkeys(['mc', 'micro', 'MC'],
                        {'layers': [2, 3, 3, 2],
                         'embed_dims': [48, 48, 96, 192, 384],
                         'strides': [2, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_rate': [0., 0., 0., 0.],
                         'drop_path_rate': 0.2,
                         'layer_scale_init_value': 1e-5}),

        **dict.fromkeys(['m', 'miny', 'M'],
                        {'layers': [2, 3, 3, 2],
                         'embed_dims': [64, 64, 128, 256, 512],
                         'strides': [2, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_rate': [0., 0., 0., 0.],
                         'drop_path_rate': 0.2,
                         'layer_scale_init_value': 1e-5}),
        
          **dict.fromkeys(['t', 'tiny', 'T'],
                        {'layers': [2, 3, 3, 2],
                         'embed_dims': [96, 96, 192, 384, 768],
                         'strides': [2, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_rate': [0., 0., 0., 0.],
                         'drop_path_rate': 0.2,
                         'layer_scale_init_value': 1e-5}),      

        **dict.fromkeys(['s', 'small', 'S'],
                        {'layers': [2, 6, 6, 2],
                         'embed_dims': [96, 96, 192, 384, 768],
                         'strides': [2, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_rate': [0., 0., 0., 0.],
                         'drop_path_rate': 0.3,
                         'layer_scale_init_value': 1e-5}),

        **dict.fromkeys(['b', 'base', 'B'],
                        {'layers': [2, 6, 6, 2],
                         'embed_dims': [128, 128, 256, 512, 1024],
                         'strides': [2, 2, 2, 2],
                         'expand_ratio': [4, 4, 4, 4],
                         'drop_rate': [0., 0., 0., 0.],
                         'drop_path_rate': 0.5,
                         'layer_scale_init_value': 1e-5})
    }

    def __init__(self, arch='tiny', num_classes=1000, norm_type='GN'):
        super().__init__()

        if isinstance(arch, str):
            assert arch in self.arch_settings, f'Unavailable arch, please choose from {set(self.arch_settings)} or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, f'The arch dict must have "layers" and "embed_dims", but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        expand_ratio = arch['expand_ratio']
        strides = arch['strides']
        drop_rate = arch['drop_rate']
        drop_path_rate = arch['drop_path_rate']
        layer_scale_init_value = arch['layer_scale_init_value']

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]  # stochastic depth decay rule

        dpr_list = split_list(dpr, layers)

        if norm_type == 'GN':
            self.norm = nn.GroupNorm(1, embed_dims[-1])
        elif norm_type == 'BN':
            self.norm = nn.BatchNorm2d(embed_dims[-1])
        else:
            raise ValueError()

        self.conv_stem = ConvStem(3, embed_dims[0])

        self.layers1 = make_layers(embed_dims[0], embed_dims[1], layers[0], expand_ratio[0], drop_rate[0], dpr_list[0],
                                   strides[0], layer_scale_init_value, 'C')

        self.layers2 = make_layers(embed_dims[1], embed_dims[2], layers[1], expand_ratio[1], drop_rate[1], dpr_list[1],
                                   strides[1], layer_scale_init_value, 'C')

        self.layers3 = make_layers(embed_dims[2], embed_dims[3], layers[2], expand_ratio[2], drop_rate[2], dpr_list[2],
                                   strides[2], layer_scale_init_value, 'T')

        self.layers4 = make_layers(embed_dims[3], embed_dims[4], layers[3], expand_ratio[3], drop_rate[3], dpr_list[3],
                                   strides[3], layer_scale_init_value, 'T')       

        self.classifier = nn.Sequential(self.norm,
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1),
                                        ) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # init for image classification
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv_stem(x)

        out = self.layers1(out)

        out = self.layers2(out)

        out = self.layers3(out)

        out = self.layers4(out)

        out = self.classifier(out).flatten(1)
        return out


def CoFM_Micro(num_classes):
    model = CoFM(arch='mc', num_classes=num_classes)
    return model


def CoFM_Miny(num_classes):
    model = CoFM(arch='m', num_classes=num_classes)
    return model


def CoFM_Tiny(num_classes):
    model = CoFM(arch='t', num_classes=num_classes)
    return model


def CoFM_Small(num_classes):
    model = CoFM(arch='s', num_classes=num_classes)
    return model


def CoFM_Base(num_classes):
    model = CoFM(arch='b', num_classes=num_classes)
    return model
