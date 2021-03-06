import torch
import torch.nn as nn


def make_pseudo_label(logits, threshold):
    pseudo_label = torch.softmax(logits, dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(threshold).float()
    return targets_u, mask


def sharpening(soft_labels, temp):
    soft_labels = soft_labels.pow(temp)
    return soft_labels / soft_labels.abs().sum(1, keepdim=True)


def tempereture_softmax(logits, tau):
    return (logits/tau).softmax(1)


def mixup(x, y, alpha):
    device = x.device
    b = x.shape[0]
    permute = torch.randperm(b)
    perm_x = x[permute]
    perm_y = y[permute]
    factor = torch.distributions.beta.Beta(alpha, alpha).sample((b,1)).to(device)
    if len(x.size())== 4:
        x_factor = factor[...,None,None]
    else:
        x_factor = factor
    mixed_x = x_factor * x + (1-x_factor) * perm_x
    mixed_y = factor * y + (1-factor) * perm_y
    return mixed_x, mixed_y


def anneal_loss(logits, labels, loss, global_step, max_iter, num_classes, schedule):
    tsa_start = 1 / num_classes
    threshold = get_tsa_threshold(
        schedule, global_step, max_iter,
        tsa_start, end=1
    )
    with torch.no_grad():
        probs = logits.softmax(1)
        correct_label_probs = probs.gather(1, labels[:,None]).squeeze()
        mask = (correct_label_probs < threshold).float()
    return (loss * mask).mean()


def get_tsa_threshold(schedule, global_step, max_iter, start, end):
    step_ratio = global_step / max_iter
    if schedule == "linear":
        coef = step_ratio
    elif schedule == "exp":
        scale = 5
        coef = ((step_ratio - 1) * scale).exp()
    elif schedule == "log":
        scale = 5
        coef = 1 - (-step_ratio * scale).exp()
    else:
        raise NotImplementedError
    return coef * (end - start) + start