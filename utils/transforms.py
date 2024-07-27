import math
from typing import Optional, Tuple, Union

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import str_to_interp_mode, ResizeKeepRatio, CenterCropOrPad, ToNumpy
from torchvision import transforms


def transforms_noaug_train(
        img_size: Union[int, Tuple[int, int]] = 224,
        interpolation: str = 'bilinear',
        use_prefetcher: bool = False,
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
):
    """ No-augmentation image transforms for training.

    Args:
        img_size: Target image size.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.

    Returns:

    """
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            )
        ]
    return transforms.Compose(tfl)


def transforms_imagenet_train(
        img_size: Union[int, Tuple[int, int]] = 224,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        use_prefetcher: bool = False,
):
    """ ImageNet-oriented image transforms for training.

    Args:
        img_size: Target image size.
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.

    Returns:
         * all data through the first (primary) transform, called the 'clean' data
         * a portion of the data through the secondary transform
         * normalizes and converts the branches above with the third, final transform
    """
    primary_tfl = [transforms.RandomResizedCrop(size=img_size)]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []

    if auto_augment:  # default using RandAugment (ra) or TrivialAugment (ta)
        assert isinstance(auto_augment, str) and auto_augment.lower() in ['ra', 'ta']
        if auto_augment.lower() == 'ra':
            secondary_tfl += [transforms.autoaugment.RandAugment(2, 15)]
        elif auto_augment.lower() == 'ta':
            secondary_tfl += [transforms.autoaugment.TrivialAugmentWide()]
        else:
            raise NotImplementedError()

    if color_jitter is not None:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        if color_jitter_prob is not None:
            secondary_tfl += [
                transforms.RandomApply([
                    transforms.ColorJitter(*color_jitter),
                ],
                    p=color_jitter_prob
                )
            ]
        else:
            secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    if grayscale_prob:
        secondary_tfl += [transforms.RandomGrayscale(p=grayscale_prob)]

    if gaussian_blur_prob:
        secondary_tfl += [
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23),  # hardcoded for now
            ],
                p=gaussian_blur_prob,
            )
        ]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            ),
        ]
        if re_prob > 0.:
            final_tfl += [
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device='cpu',
                )
            ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_imagenet_eval(
        img_size: Union[int, Tuple[int, int]] = 224,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        use_prefetcher: bool = False,
):
    """ ImageNet-oriented image transform for evaluation and inference.

    Args:
        img_size: Target image size.
        crop_pct: Crop percentage. Defaults to 0.875 when None.
        crop_mode: Crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.

    Returns:
        Composed transform pipeline
    """
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        scale_size = tuple([math.floor(x / crop_pct) for x in img_size])
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    tfl = []

    if crop_mode == 'squash':
        # squash mode scales each edge to 1/pct of target, then crops
        # aspect ratio is not preserved, no img lost if crop_pct == 1.0
        tfl += [
            transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
            transforms.CenterCrop(img_size),
        ]
    elif crop_mode == 'border':
        # scale the longest edge of image to 1/pct of target edge, add borders to pad, then crop
        # no image lost if crop_pct == 1.0
        fill = [round(255 * v) for v in mean]
        tfl += [
            ResizeKeepRatio(scale_size, interpolation=interpolation, longest=1.0),
            CenterCropOrPad(img_size, fill=fill),
        ]
    else:
        # default crop model is center
        # aspect ratio is preserved, crops center within image, no borders are added, image is lost
        if scale_size[0] == scale_size[1]:
            # simple case, use torchvision built-in Resize w/ the shortest edge mode (scalar size arg)
            tfl += [
                transforms.Resize(scale_size[0], interpolation=str_to_interp_mode(interpolation))
            ]
        else:
            # resize the shortest edge to matching target dim for non-square target
            tfl += [ResizeKeepRatio(scale_size)]
        tfl += [transforms.CenterCrop(img_size)]

    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            )
        ]

    return transforms.Compose(tfl)


def create_transform(
        input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
        is_training: bool = False,
        no_aug: bool = False,
        hflip: float = 0.5,
        vflip: float = 0.,
        color_jitter: Union[float, Tuple[float, ...]] = 0.4,
        color_jitter_prob: Optional[float] = None,
        grayscale_prob: float = 0.,
        gaussian_blur_prob: float = 0.,
        auto_augment: Optional[str] = None,
        interpolation: str = 'bilinear',
        mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        crop_pct: Optional[float] = None,
        crop_mode: Optional[str] = None,
        use_prefetcher: bool = False,
):
    """

    Args:
        input_size: Target input size (channels, height, width) tuple or size scalar.
        is_training: Return training (random) transforms.
        no_aug: Disable augmentation for training (useful for debug).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        interpolation: Image interpolation mode.
        auto_augment: Auto augment configuration string.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        crop_pct: Inference crop percentage (output size / resize size).
        crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        use_prefetcher: Pre-fetcher enabled. Do not convert image to tensor or normalize.

    Returns:
        Composed transforms
    """
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and no_aug:
        transform = transforms_noaug_train(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
        )
    elif is_training:
        transform = transforms_imagenet_train(
            img_size,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            gaussian_blur_prob=gaussian_blur_prob,
            auto_augment=auto_augment,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            use_prefetcher=use_prefetcher,
        )
    else:
        transform = transforms_imagenet_eval(
            img_size,
            interpolation=interpolation,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
            crop_mode=crop_mode,
            use_prefetcher=use_prefetcher,
        )

    return transform
