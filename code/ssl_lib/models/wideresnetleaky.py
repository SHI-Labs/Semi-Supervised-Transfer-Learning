#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
import torch.nn as nn

#from torch.nn import BatchNorm2d
from .utils import BatchNorm2d, BaseModel

'''
    As in the paper, the wide resnet only considers the resnet of the pre-activated version, 
    and it only considers the basic blocks rather than the bottleneck blocks.
'''


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, drop_rate=0, stride=1, pre_res_act=False,bn_momentum=0.001):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = BatchNorm2d(out_chan, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.dropout = nn.Dropout(drop_rate) if not drop_rate == 0 else None
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=True
            )
        self.pre_res_act = pre_res_act
        # self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu1(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        if self.dropout is not None:
            residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = act1 if self.pre_res_act else x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out = shortcut + residual
        return out

    def init_weight(self):
        # for _, md in self.named_modules():
        #     if isinstance(md, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #         if md.bias is not None:
        #             nn.init.constant_(md.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class WideResnet_leaky(nn.Module):
    def __init__(self, k=2, n=28, drop_rate=0, n_classes=10,bn_momentum=0.001):
        super(WideResnet_leaky, self).__init__()
        self.k, self.n = k, n
        assert (self.n - 4) % 6 == 0
        n_blocks = (self.n - 4) // 6
        n_layers = [16, ] + [self.k * 16 * (2 ** i) for i in range(3)]

        self.conv1 = nn.Conv2d(3, n_layers[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self.create_layer(n_layers[0], n_layers[1], bnum=n_blocks, stride=1,
                                        drop_rate=drop_rate, pre_res_act=True,bn_momentum=bn_momentum)
        self.layer2 = self.create_layer(n_layers[1], n_layers[2], bnum=n_blocks, stride=2,
                                        drop_rate=drop_rate, pre_res_act=False,bn_momentum=bn_momentum)
        self.layer3 = self.create_layer(n_layers[2], n_layers[3], bnum=n_blocks, stride=2,
                                        drop_rate=drop_rate, pre_res_act=False,bn_momentum=bn_momentum)
        self.bn_last = BatchNorm2d(n_layers[3], momentum=bn_momentum)
        self.relu_last = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64 * self.k, n_classes, bias=True)

        self.init_weight()


    def create_layer(self, in_chan, out_chan, bnum, stride=1, drop_rate=0, pre_res_act=False,bn_momentum=0.001):
        layers = [
            BasicBlockPreAct(
                in_chan,
                out_chan,
                drop_rate=drop_rate,
                stride=stride,
                pre_res_act=pre_res_act,bn_momentum=bn_momentum), ]
        for _ in range(bnum - 1):
            layers.append(
                BasicBlockPreAct(
                    out_chan,
                    out_chan,
                    drop_rate=drop_rate,
                    stride=1,
                    pre_res_act=False, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x, return_fmap=False):
        feat = self.conv1(x)
        ret1 = feat
        feat = self.layer1(feat)
        ret2 = feat
        feat = self.layer2(feat)  # 1/2
        ret3 = feat
        feat = self.layer3(feat)  # 1/4
        ret4 = feat

        feat = self.bn_last(feat)
        feat = self.relu_last(feat)
        ret5 = feat
        #feat = torch.mean(torch.mean(feat, dim=-1), dim=-1)
        feat = self.pool(feat).squeeze()
        ret6 = feat
        feat = self.classifier(feat)
        if return_fmap:
            return [ret1, ret2, ret3, ret4, ret5,ret6, feat]
        else:
            return feat 

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, BatchNorm2d):
                m.update_batch_stats = flag


def wideresnetleaky(depth, widen_factor, drop_rate, num_classes,bn_momentum=0.001):
    """
    Constructs a WideResNet model.
    """
    return WideResnet_leaky(k=widen_factor, n=depth, drop_rate=drop_rate, n_classes=num_classes,bn_momentum=bn_momentum)


if __name__ == "__main__":
    import random
    import torch
    import numpy as np
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    x = torch.randn(2, 3, 32, 32)

    net = WideResnet_leaky(k=2, n=28, n_classes=10)
    net = tmp(k=2, n=28, drop_rate=0, n_classes=10)
    out = net(x)
    print(out)
