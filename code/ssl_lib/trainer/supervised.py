import os, numpy, random, time, math
import torch
import torch.nn.functional as F
from ssl_lib.utils import  Bar, AverageMeter


def supervised_train(epoch,train_loader, model,optimizer,lr_scheduler, cfg,device):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    labeled_acc = AverageMeter()

    n_iter = cfg.n_imgs_per_epoch // cfg.l_batch_size
    bar = Bar('Supervised Training', max=n_iter)
    
    end = time.time()
    for batch_idx, (data_l, _) in enumerate(train_loader):
        inputs_l, labels = data_l
        inputs_l, labels = inputs_l.to(device), labels.to(device)
        data_time.update(time.time() - end)

        bs = inputs_l.size(0)
        cur_iteration = epoch*cfg.per_epoch_steps+batch_idx

        logits_l = model(inputs_l)

        loss = F.cross_entropy(logits_l, labels)     

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc_l = (logits_l.max(1)[1] == labels).float().mean()

        losses.update(loss.item())
        labeled_acc.update(acc_l.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (batch_idx+1) % 10==0:
            bar.suffix = ("{batch:4}/{iter:4}. LR:{lr:.6f}. Data:{dt:.3f}s. Batch:{bt:.3f}s. Loss:{loss:.4f}. Acc_L:{acc_l:.4f}.".format(
                    batch=batch_idx+1,
                    iter=n_iter,
                    lr=cur_lr,
                    dt=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    acc_l=labeled_acc.avg))
            bar.next()
    
    bar.finish()

    return losses.avg, labeled_acc.avg
