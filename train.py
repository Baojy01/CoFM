import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from timm.utils import ModelEmaV2, get_state_dict, unwrap_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from timm.models import load_checkpoint

from models import config, get_model
from utils import GetData, train_runner, val_runner

model_names = config.models
parser = argparse.ArgumentParser(description='PyTorch image training')

# Model parameters
parser.add_argument('--seed', default=42, type=int, help='Seed for initializing training')
parser.add_argument('--arch', default='CoFM_Miny', choices=model_names, metavar='ARCH', help='Models architecture')
parser.add_argument('--dataset', default='imagenet1k', choices=['imagenet1k', 'cifar10', 'cifar100'], help='Dataset for training')
parser.add_argument('--data-dir', default='../autodl-tmp/ImageNet', help='Path to dataset')
parser.add_argument('--num-classes', default=1000, type=int, help='Number of classes for classification')
parser.add_argument('--input-size', default=224, type=int, help='Input image size')

parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='Mini-batch size (default: 1024), this is the total batch size of all GPUs on the current node when using DP or DDP')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='Number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='Number of total epochs to run (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')

# Learning rate schedule parameters, default: CosineAnnealingLR
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='Initial learning rate (default: 0.001)')
parser.add_argument('--warmup', default=True, type=bool, help='Whether using warmup or not (default: True)')
parser.add_argument('--warmup-epoch', default=5, type=int, metavar='N', help='Use warmup epoch number (default: 5)')

# Augmentation parameters
parser.add_argument('--aa', type=str, default='TA', metavar='NAME', help='Use AutoAugment policy (default: TrivialAugment)'),
parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability (default: 0.5)')
parser.add_argument('--vflip', type=float, default=0., help='Vertical flip training aug probability (default: 0)')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
parser.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT', help='Probability of random color jitter (default: None)')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

# Optimizer parameters, default Optimizer: AdamW
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Optimizer momentum')
parser.add_argument('--weight-decay', default=0.05, type=float, metavar='W', help='Weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')

# Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase probability (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False, help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=True, help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998, help='decay factor for model weights moving average (default: 0.9998)')

# Mixup params
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5, help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

#DistributedDataParallel params
parser.add_argument('--ddp', default=True, type=bool, help='use DistributedDataParallel for training')
parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--sync-bn', default=True, type=bool, dest='sync_bn', help='Use sync batch norm')
parser.add_argument('--amp', default=True, type=bool, help='Use torch.cuda.amp for mixed precision training')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=False, type=bool, help='directory if using pre-trained models')
args = parser.parse_args()

def cleanup():
    torch.distributed.destroy_process_group()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def run(local_rank, args):
    
    args.local_rank = local_rank
    
    if args.seed is not None:
        set_seed(args.seed)

    results_dir = './ablation_results5/%s/' % (args.arch + '_' + args.dataset + '_bs' + str(args.batch_size) + '_epochs_' + str(args.epochs))
    save_dir = './ablation_checkpoint5/%s/' % (args.arch + '_' + args.dataset + '_bs' + str(args.batch_size) + '_epochs_' + str(args.epochs))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:23456', rank=args.local_rank, world_size=args.nprocs)      
        
    # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs   
    args.batch_size = args.batch_size // args.nprocs

    train_set, val_set = GetData(args)
    
    if args.ddp:
        train_sampler  = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        val_sampler = torch.utils.data.SequentialSampler(val_set)
        
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    model = get_model(args.arch, args.num_classes)
    model = model.to(device)

    mixup_fn = None
    mixup_active = args.mixup > 0. or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
            
    
    if args.ddp and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.ddp and args.nprocs > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)  
        model_without_ddp = model.module  
    

    log_dir = os.path.join(save_dir, "last_model.pth")
    train_dir = os.path.join(results_dir, "train.csv")
    val_dir = os.path.join(results_dir, "val.csv")

    best_top1, best_top5 = 0.0, 0.0
    Loss_train = []
    Loss_val, Accuracy_val_top1, Accuracy_val_top5 = [], [], []

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
        elif os.path.isfile(log_dir):
            print("There is no checkpoint found at '{}', then loading default checkpoint at '{}'.".format(args.resume, log_dir))
            checkpoint = torch.load(log_dir, map_location='cuda:{}'.format(args.local_rank))  # default load last_model.pth
        else:
            raise FileNotFoundError()

        if args.model_ema:
            model_without_ddp.load_state_dict(checkpoint['model_ema'])
        else:
            model_without_ddp.load_state_dict(checkpoint['model'])
            
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        if args.start_epoch < args.epochs:
            best_top1, best_top5 = checkpoint['best_top1'], checkpoint['best_top5']
            print('Loading model successfully, current start_epoch={}.'.format(args.start_epoch))
            trainF = open(train_dir, 'a+')
            valF = open(val_dir, 'a+')
        else:
            raise ValueError('epochs={}, but start_epoch={} in the saved model, please reset epochs larger!'.format(args.epochs, args.start_epoch))
    else:
        trainF = open(results_dir + 'train.csv', 'w')
        valF = open(results_dir + 'val.csv', 'w')
        trainF.write('{},{}\n'.format('epoch', 'loss'))
        valF.write('{},{},{},{},{},{}\n'.format('epoch', 'val_loss', 'val_top1', 'val_top5', 'best_top1', 'best_top5'))

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.ddp:
            train_sampler.set_epoch(epoch)
            
        torch.cuda.synchronize()
        time_star = time.time()
        loss = train_runner(model, device, train_loader, criterion, optimizer, lr_scheduler, args, epoch, 
                            scaler=scaler, model_ema=model_ema, mixup_fn=mixup_fn)
        
        val_top1, val_top5, val_loss = val_runner(model, device, val_loader, args)
        
        torch.cuda.synchronize()
        time_end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if args.warmup:
            if epoch >= args.warmup_epoch:
                lr_scheduler.step()
        else:
            lr_scheduler.step()

        # save weights
        save_files = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_top1': best_top1,
            'best_top5': best_top5,
            'epoch': epoch}
        if args.amp:
            save_files['scaler'] = scaler.state_dict() 
        if args.model_ema:
            save_files['model_ema'] = get_state_dict(model_ema, unwrap_model)

        if args.local_rank == 0:
            torch.save(save_files, log_dir)

        if best_top5 < val_top5:
            best_top5 = val_top5

        if best_top1 < val_top1:
            best_top1 = val_top1
            
            if args.local_rank == 0:
                torch.save(save_files, os.path.join(save_dir, "best_model.pth"))

        Loss_train.append(loss)

        Loss_val.append(val_loss)
        Accuracy_val_top1.append(val_top1)
        Accuracy_val_top5.append(val_top5)
        
        if args.local_rank == 0:
            print("Train Epoch: {} \t train loss: {:.4f}".format(epoch, loss))
            print("val_loss: {:.4f}, val_top1: {:.4f}%, val_top5: {:.4f}%".format(val_loss, val_top1, val_top5))
            print("best_val_top1: {:.4f}%, best_val_top5: {:.4f}%, lr: {:.6f}".format(best_top1, best_top5, lr))
            print('Each epoch running time: {:.4f} s'.format(time_end - time_star))
            trainF.write('{},{},{}\n'.format(epoch, loss, lr))
            valF.write('{},{},{},{},{},{}\n'.format(epoch, val_loss, val_top1, val_top5, best_top1, best_top5))

        trainF.flush()
        valF.flush()

    trainF.close()
    valF.close()
    
    cleanup()

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    args.nprocs = torch.cuda.device_count()
    print('use {} gpus!'.format(args.nprocs))
    # args.lr =  args.lr * args.nprocs
    mp.spawn(run, nprocs=args.nprocs, args=(args, ))
    
    print('Finished Training!')
    
if __name__ == "__main__":
    main()