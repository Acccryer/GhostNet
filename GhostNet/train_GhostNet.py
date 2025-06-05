# 2025.06.05 - Training script for GhostNet on ImageNet
#            Adapted from validate.py and ghostnet.py
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ghostnet import ghostnet
from validate import AverageMeter, accuracy  # validate.py中的工具函数

torch.backends.cudnn.benchmark = True

# 命令行参数
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training for GhostNet')
parser.add_argument('--data', metavar='DIR', default='/home/00/imagenet',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='/cache/models/',
                    help='path to output files')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--width', type=float, default=1.0,
                    help='Width ratio (default: 1.0)')
parser.add_argument('--dropout', type=float, default=0.2, metavar='PCT',
                    help='Dropout rate (default: 0.2)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

def main():
    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    # 创建模型
    model = ghostnet(num_classes=args.num_classes, width=args.width, dropout=args.dropout)
    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    elif args.num_gpu == 1:
        model = model.cuda()
    else:
        model = model  # CPU

    # 加载检查点（如果提供）
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info(f"=> loaded checkpoint '{args.resume}'")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")

    # 数据预处理
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda() if args.num_gpu > 0 else nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_acc1 = 0
    for epoch in range(args.epochs):
        # 训练一个 epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # 验证
        metrics = validate(val_loader, model, criterion, args)
        acc1 = metrics['top1']

        # 保存检查点
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.output_dir)

        # 更新学习率
        scheduler.step()
        logging.info(f'Epoch {epoch + 1}/{args.epochs} - Best Acc@1: {best_acc1:.4f}')

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if args.num_gpu > 0:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # 前向传播
        output = model(input)
        loss = criterion(output, target)

        # 计算准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新指标
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), output.size(0))
        top5.update(acc5.item(), output.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                         f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                         f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                         f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.num_gpu > 0:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), output.size(0))
            top5.update(acc5.item(), output.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info(f'Test: [{i}/{len(val_loader)}] '
                             f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                             f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                             f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                             f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    metrics = {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg}
    logging.info(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return metrics

def save_checkpoint(state, is_best, output_dir):
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, os.path.join(output_dir, 'model_best.pth.tar'))

if __name__ == '__main__':
    main()