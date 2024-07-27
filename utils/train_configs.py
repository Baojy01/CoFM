import torch
import math
import sys
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from prefetch_generator import BackgroundGenerator


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    acc = [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(models):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


def adjust_learning_rate(optimizer, lr_scheduler, args, epoch, nBatch, batch):
    warmup_epoch = args.warmup_epoch
    warmup = args.warmup
    if epoch < warmup_epoch and warmup is True:
        warmup_steps = nBatch * args.warmup_epoch
        lr_step = args.lr / (warmup_steps - 1)
        current_step = epoch * nBatch + batch
        lr = lr_step * current_step
    else:
        lr = lr_scheduler.get_last_lr()[0]

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_runner(model, mydevice, trainloader, criterion, optimizer, lr_scheduler, args, epoch, scaler=None, model_ema=None, mixup_fn=None):
    losses = AverageMeter()
    model.train()

    nBatch = len(trainloader)

    for i, (inputs, labels) in BackgroundGenerator(enumerate(trainloader)):
        adjust_learning_rate(optimizer, lr_scheduler, args, epoch, nBatch, i)
        
        inputs, labels = inputs.to(mydevice), labels.to(mydevice)
        
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
            
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if not math.isfinite(loss.item()) or math.isnan(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        torch.cuda.synchronize() #torch.distributed.barrier()        
        if args.ddp:
            loss = reduce_mean(loss, args.nprocs)
            
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            
            if args.clip_grad is not None:
                clip_grad_norm(model.parameters(), max_norm=args.clip_grad, norm_type=2)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            if args.clip_grad is not None:
                clip_grad_norm(model.parameters(), max_norm=args.clip_grad, norm_type=2)
                
            optimizer.step()
            
        torch.cuda.synchronize()   
        if model_ema is not None:
            model_ema.update(model)

    return losses.avg


def val_runner(model, mydevice, val_loader, args):
    
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, (inputs, labels) in BackgroundGenerator(enumerate(val_loader)):
            inputs, labels = inputs.to(mydevice), labels.to(mydevice)

            output = model(inputs)
            loss = criterion(output, labels)

            prec1, prec5 = accuracy(output, labels, topk=(1, 5))
            
            torch.cuda.synchronize()
            if args.ddp:
                loss = reduce_mean(loss, args.nprocs)
                prec1 = reduce_mean(prec1, args.nprocs)
                prec5 = reduce_mean(prec5, args.nprocs)
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    return top1.avg, top5.avg, losses.avg
