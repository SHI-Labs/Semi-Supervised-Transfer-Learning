import os
import numpy as np
import torch

from .wideresnetleaky import wideresnetleaky
from .resnet import  build_ResNet

def gen_model(arch,depth,widen_factor, num_classes,pretrained_weight_path='',pretrained=False,bn_momentum=0.001):
    print("==> creating model '{}'".format(arch))
    orig_num_classes = 1000 if pretrained else num_classes
    if arch.endswith('wideresnetleaky'):
        model = wideresnetleaky(depth=depth,
                               widen_factor=widen_factor,
                               drop_rate=0,
                               num_classes=orig_num_classes,
                               bn_momentum=bn_momentum)
    elif arch.startswith('resnet'):
        model = build_ResNet(depth=depth,num_classes=orig_num_classes)
    else:
        raise NotImplementedError
    if pretrained:
        net_name = f"{arch}_{depth}_{widen_factor}"
        ckpt_path = os.path.join(pretrained_weight_path,net_name+'.pth')
        print('load pretrained model from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
        if orig_num_classes!= num_classes:
            if arch.endswith('wideresnetleaky'):
                nChannel = 64*widen_factor
                model.classifier = torch.nn.Linear(nChannel, num_classes)
                torch.nn.init.xavier_normal_(model.classifier.weight)
                torch.nn.init.constant_(model.classifier.bias, 0)
            else:
                expansion =1 if depth<50 else 4
                feat_dim = 512 * expansion # 2048
                model.fc = torch.nn.Linear(feat_dim, num_classes)
                torch.nn.init.xavier_normal_(model.fc.weight)
                torch.nn.init.constant_(model.fc.bias, 0)
    return model