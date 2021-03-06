import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(y, target, mask=None):
    if len(target.shape)<2: #if target.ndim == 1: # for hard label
        loss = F.cross_entropy(y, target, reduction="none")
    else:
        loss = -(target * F.log_softmax(y, 1)).sum(1)
    if mask is not None:
        loss = loss * mask
    return loss.mean()

class CrossEntropy(nn.Module):
    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask)
