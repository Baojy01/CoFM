from torchvision import datasets, transforms
from .transforms import create_transform


def GetData(args):
    if args.dataset.lower() == 'cifar10':
        transform_trains = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vals = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_sets = datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=transform_trains)
        val_sets = datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform_vals)
    elif args.dataset.lower() == 'cifar100':
        transform_trains = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vals = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_sets = datasets.CIFAR100(root="./data/CIFAR100", train=True, download=True, transform=transform_trains)
        val_sets = datasets.CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=transform_vals)

    elif args.dataset.lower() == 'imagenet1k':
        transform_trains = create_transform(
            input_size=args.input_size,
            is_training=True,
            hflip=args.hflip,
            vflip=args.vflip,
            auto_augment=args.aa,
            color_jitter=args.color_jitter,
            color_jitter_prob=args.color_jitter_prob,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        transform_vals = create_transform(input_size=args.input_size)
        train_sets = datasets.ImageFolder(root=args.data_dir+"/train", transform=transform_trains)
        val_sets = datasets.ImageFolder(root=args.data_dir+"/val", transform=transform_vals)
    else:
        raise NotImplementedError()

    return train_sets, val_sets
