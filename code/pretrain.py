from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from ssl_lib.models.builder import gen_model
from ssl_lib.datasets.dataset_class import ImageNet32
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader

from ssl_lib.utils import Bar, Logger, AverageMeter, accuracy, init_trial_path
print('pid:',os.getpid())

parser = argparse.ArgumentParser(description='PyTorch Pretraining')
# Datasets
parser.add_argument("--model", default="wideresnetleaky", type=str, help="model architecture") #resner or wideresnetleaky
parser.add_argument("--depth", default=28,  type=int, help="model depth")
parser.add_argument("--widen_factor", default=2,  type=int, help="widen factor for wide resnet 2 for cifar10, 8 for cifar100")
parser.add_argument('-d', '--dataset', default='imagenet32', type=str)
parser.add_argument('--data_root', default='image-net/imagenet32', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
parser.add_argument('--whiten_image', default=1, type=int, help='normalize or not image')
# Optimizer
parser.add_argument('--epochs', default=80, type=int, help='number of total epochs to run')
parser.add_argument('--train_batch', default=256, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test_batch', default=512, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4 on cifar)')
parser.add_argument('--decay_step', type=int, default=[30, 60], nargs='+')
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--nesterov', action='store_true', default=True, help='use nesterov momentum')
# Checkpoint
parser.add_argument('--out_dir',default='results/pretrained', 
                    help='directory to output the result')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--num_save_epoch', type=int, default=10,help='save model frequency')
args = parser.parse_args()
args.net_name = f"{args.model}_{args.depth}_{args.widen_factor}"
args.task_name = f"pretrain_{args.net_name}@{args.dataset}"
args = init_trial_path(args)
print(args)
cudnn.benchmark = True

def main():
    print("***** Running training *****")
    print(f"  Task = {args.dataset}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Total train batch size = {args.train_batch}")

    trainset, testset, transform_train, transform_test, num_classes = init_data()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,
                                            num_workers=args.workers, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers, drop_last=False, pin_memory=True)

    model = gen_model(args.model, args.depth, args.widen_factor, num_classes, '',False,0.1)
    model = model.cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        # if len(param.size()) == 1:
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias, conv2d.bias
            # print(name)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params, 'weight_decay': args.weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
    optimizer = optim.SGD(param_list, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    schedular = MultiStepLR(optimizer,args.decay_step, args.lr_decay) 

    # train the model from scratch
    best_acc = 0
    start_epoch = 0
    # Resume
    title = args.task_name
    log_names = ['Train Loss', 'Valid Loss', 'Train Acc.','Valid Acc.', 'Train Top5','Valid Top5']
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt)
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.save_path, 'log.txt'), title=title)
        logger.set_names(log_names)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        train_loss, train_acc, train_top5 = train(args, trainloader, model,  optimizer)

        test_loss, test_acc, test_top5 = test(testloader, model)
        log_vals = [train_loss, test_loss, train_acc, test_acc,train_top5, test_top5]
        logger.append(log_vals)
        if test_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.net_name}_best.pth'))

        if epoch%args.num_save_epoch==0 or epoch==args.epochs-1:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'{args.net_name}_{epoch}.pth'))
        
        schedular.step()
        best_acc = max(test_acc, best_acc)
        log_str=f"Epoch: {epoch},"
        for k,v in zip(log_names, log_vals):
            log_str += f"{k}: {v},"
        print(log_str)
    print('Best test acc:', best_acc)
    return model, best_acc


def init_data():
    """
    setup all kinds of constants here, just to make it cleaner :)
    """

    if args.dataset=='imagenet32':
        mean = (0.4811, 0.4575, 0.4078)
        std = (0.2605 , 0.2533, 0.2683)
        num_classes = 1000
    else:
        raise NotImplementedError

    if args.whiten_image==0:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # with p = 0.5
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),  # with p = 1
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = ImageNet32(root=args.data_root, train=True,transform=transform_train)
    testset = ImageNet32(root=args.data_root, train=False,transform=transform_test)


    return trainset, testset, transform_train, transform_test, num_classes


def train(args,trainloader,model,optimizer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time

        inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets, reduction='mean')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        optimizer.zero_grad()
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        cur_lr = optimizer.param_groups[0]['lr']
        bar.suffix = '({batch}/{size}) Time: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | LR: {LR:.6f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            bt=batch_time.avg,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
            LR=cur_lr,
        )
        bar.next()

    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test(testloader, model):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            inputs, targets = inputs.cuda(), targets.cuda()
            # compute output
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='mean')

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Time: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    main()
