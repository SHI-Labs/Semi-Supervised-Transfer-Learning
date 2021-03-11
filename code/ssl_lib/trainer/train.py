
import os, numpy, random, time, math
import torch
import torch.nn.functional as F

from ssl_lib.algs import utils as alg_utils
from ssl_lib.models import utils as model_utils
from ssl_lib.consistency.builder import gen_consistency
from ssl_lib.consistency.regularizer import Distribution_Loss
from ssl_lib.param_scheduler import scheduler
from ssl_lib.utils import  Bar, AverageMeter
from .supervised import supervised_train


LABELED_FEAT_TABLES=None
UNLABELED_FEAT_TABLES=None

def get_mask(logits,threshold, num_class=10):
    ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    threshold = threshold * math.log(num_class)
    mask = ent.le(threshold).float()
    return mask

def update_feat_table(cur_feat_l,cur_feat_u,feat_table_size_l=-1,feat_table_size_u=-1,mask_l=None, mask_u=None):
    global LABELED_FEAT_TABLES,UNLABELED_FEAT_TABLES
    if mask_l is not None:
        mask_l = mask_l.nonzero().flatten()
        mask_u = mask_u.nonzero().flatten()
        cur_feat_l=cur_feat_l[mask_l]
        cur_feat_u=cur_feat_u[mask_u]
    if feat_table_size_l>0:
        if LABELED_FEAT_TABLES is None:
            LABELED_FEAT_TABLES = cur_feat_l
            UNLABELED_FEAT_TABLES = cur_feat_u
        else:
            LABELED_FEAT_TABLES = torch.cat([LABELED_FEAT_TABLES,cur_feat_l])
            UNLABELED_FEAT_TABLES = torch.cat([UNLABELED_FEAT_TABLES,cur_feat_u])
            if len(LABELED_FEAT_TABLES) > feat_table_size_l:
                LABELED_FEAT_TABLES = LABELED_FEAT_TABLES[-feat_table_size_l:]
            if len(UNLABELED_FEAT_TABLES) > feat_table_size_u:
                UNLABELED_FEAT_TABLES = UNLABELED_FEAT_TABLES[-feat_table_size_u:]
        feat_l = LABELED_FEAT_TABLES
        feat_u = UNLABELED_FEAT_TABLES
        LABELED_FEAT_TABLES=LABELED_FEAT_TABLES.detach()
        UNLABELED_FEAT_TABLES=UNLABELED_FEAT_TABLES.detach()
    else:
        feat_l = cur_feat_l
        feat_u = cur_feat_u
    
    return feat_l, feat_u

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def infer_interleave(forward_fn, inputs_train,cfg,bs):
    merge_one_batch = cfg.merge_one_batch
    if cfg.interleave:
        inputs_interleave = list(torch.split(inputs_train, bs))
        inputs_interleave = interleave(inputs_interleave, bs)
        if merge_one_batch:
            inputs_interleave = [torch.cat(inputs_interleave)] ####
    else:
        inputs_interleave = [inputs_train]

    out_lists_inter = [forward_fn(inputs_interleave[0],return_fmap=True)]
    for inputs in inputs_interleave[1:]:
        out_lists_inter.append(forward_fn(inputs,return_fmap=True))

    for ret_id in [-1,-3]:
        ret_list=[]
        for o_list in out_lists_inter:
            ret_list.append(o_list[ret_id])
        # put interleaved samples back
        if cfg.interleave:
            if merge_one_batch:
                ret_list = list(torch.split(ret_list[0], bs))
            ret_list = interleave(ret_list, bs)
            feat_l = ret_list[0]
            feat_u_w, feat_u_s = torch.cat(ret_list[1:],dim=0).chunk(2)
            #feat_l,feat_u_w, feat_u_s = ret_list
        else:
            feat_l = ret_list[0][:bs]
            feat_u_w, feat_u_s = ret_list[0][bs:].chunk(2)
        if ret_id==-1:
            logits_l,logits_u_w, logits_u_s = feat_l,feat_u_w, feat_u_s
        else:
            cur_feat_l = feat_l
            cur_feat_u = feat_u_w
            cur_feat_s = feat_u_s
            feat_target = torch.cat((feat_l, feat_u_w), dim=0)
    return  logits_l,logits_u_w, logits_u_s,cur_feat_l,cur_feat_u,cur_feat_s,feat_target

def train(epoch,train_loader , model,source_model,ema_teacher,optimizer,lr_scheduler, ssl_alg, cfg,device):
    if cfg.coef==0 and cfg.lambda_mmd==0 and cfg.lambda_kd==0:
        loss, acc = supervised_train(epoch,train_loader, model,optimizer,lr_scheduler, cfg,device)
        return (loss, loss, 0, 0, 0,acc, 0,0,0,0)
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_ssl = AverageMeter()
    losses_mmd = AverageMeter()
    losses_kd = AverageMeter()
    masks_ssl = AverageMeter()
    masks_mmd = AverageMeter()
    masks_kd = AverageMeter()
    labeled_acc = AverageMeter()
    unlabeled_acc = AverageMeter()

    mmd_criterion = Distribution_Loss(loss='mmd').to(device)
    kd_criterion = Distribution_Loss(loss='mse').to(device)
    consistency_criterion = gen_consistency(cfg.consistency, use_onehot=(cfg.alg == 'pl'),
	                                        num_classes=cfg.num_classes).to(device)
    n_iter = cfg.n_imgs_per_epoch // cfg.l_batch_size
    bar = Bar('Training', max=n_iter)
    
    end = time.time()
    for batch_idx, (data_l, data_u) in enumerate(train_loader):
        inputs_l, labels = data_l
        inputs_l, labels = inputs_l.to(device), labels.to(device)
        inputs_u_w, inputs_u_s, labels_u = data_u
        inputs_u_w, inputs_u_s, labels_u = inputs_u_w.to(device), inputs_u_s.to(device), labels_u.to(device)
        inputs_train = torch.cat((inputs_l, inputs_u_w, inputs_u_s), dim=0)
        data_time.update(time.time() - end)

        bs = inputs_l.size(0)
        cur_iteration = epoch*cfg.per_epoch_steps+batch_idx

        forward_fn = model.forward
        '''
        ret_list = forward_fn(inputs_train, return_fmap=True)
        logits_l = ret_list[-1][:bs]
        logits_u_w, logits_u_s = ret_list[-1][bs:].chunk(2)
        feat_l = ret_list[-3][:bs]
        feat_u_w, feat_u_s = ret_list[-3][bs:].chunk(2)
        feat_target = torch.cat((feat_l, feat_u_w), dim=0)
        '''
        logits_l,logits_u_w, logits_u_s,feat_l,feat_u_w,feat_u_s,feat_target  = infer_interleave(forward_fn, inputs_train,cfg, bs)
        L_supervised = F.cross_entropy(logits_l, labels)

        # calc total loss
        coef = scheduler.linear_warmup(cfg.coef, cfg.warmup_iter, cur_iteration+1)
        L_consistency = torch.zeros_like(L_supervised) 
        mask = torch.zeros_like(L_supervised)   
        if cfg.coef > 0:
            # get target values
            if ema_teacher is not None and cfg.ema_teacher_train: # get target values from teacher model
                ema_forward_fn = ema_teacher.forward
                ema_logits = ema_forward_fn(inputs_train)
                ema_logits_u_w, _ = ema_logits[bs:].chunk(2)
            else:
                ema_forward_fn = forward_fn
                ema_logits_u_w = logits_u_w

            # calc consistency loss
            model.module.update_batch_stats(False)
            y, targets, mask = ssl_alg(
                stu_preds = logits_u_s,
                tea_logits = ema_logits_u_w.detach(),
                w_data = inputs_u_w,
                s_data = inputs_u_s,
                stu_forward = forward_fn,
                tea_forward = ema_forward_fn
            )
            model.module.update_batch_stats(True)
            L_consistency = consistency_criterion(y, targets, mask)

        
        L_mmd = torch.zeros_like(L_supervised) 
        mmd_mask_u = torch.zeros_like(L_supervised)   
        if cfg.lambda_mmd>0:
            mmd_mask_l = get_mask(logits_l,cfg.mmd_threshold,  num_class=cfg.num_classes)
            mmd_mask_u = get_mask(logits_u_w,cfg.mmd_threshold,  num_class=cfg.num_classes)
            if mmd_mask_l.sum()>0 and mmd_mask_u.sum()>0:
                cur_feat_l, cur_feat_u = update_feat_table(feat_l,feat_u_w,cfg.mmd_feat_table_l,cfg.mmd_feat_table_u, mask_l=mmd_mask_l, mask_u=mmd_mask_u)
                if cur_iteration>cfg.reg_warmup and len(cur_feat_l)>20:
                    L_mmd = mmd_criterion(cur_feat_l, cur_feat_u)

        L_kd = torch.zeros_like(L_supervised)
        kd_mask = torch.zeros_like(L_supervised)   
        if cfg.lambda_kd>0:
            src_inputs = torch.cat((inputs_l, inputs_u_w), dim=0)
           
            with torch.no_grad():
                src_out_list = source_model(src_inputs,return_fmap=True)
            src_logits = src_out_list[-1]
            kd_mask = get_mask(src_logits,cfg.kd_threshold, num_class=1000)
            feat_source = src_out_list[-3].detach()
            L_kd = kd_criterion(feat_target, feat_source,mask=kd_mask, reduction='none')
            
            del src_out_list,  src_inputs
    
        lambda_mmd = scheduler.linear_warmup(cfg.lambda_mmd, cfg.reg_warmup_iter, cur_iteration+1)
        lambda_kd = scheduler.linear_warmup(cfg.lambda_kd, cfg.reg_warmup_iter, cur_iteration+1)
        loss = L_supervised + coef * L_consistency + lambda_mmd * L_mmd + lambda_kd * L_kd

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=5)
        optimizer.step()
        lr_scheduler.step()

        if cfg.ema_teacher:
            model_utils.ema_update(
                ema_teacher, model, cfg.ema_teacher_factor,
                cfg.weight_decay * cur_lr if cfg.ema_apply_wd else None, 
                cur_iteration if cfg.ema_teacher_warmup else None,
                ema_train=cfg.ema_teacher_train)
        # calculate accuracy for labeled data
        acc_l = (logits_l.max(1)[1] == labels).float().mean()
        acc_ul = (logits_u_w.max(1)[1] == labels_u).float().mean()

        losses.update(loss.item())
        losses_ce.update(L_supervised.item())
        losses_ssl.update(L_consistency.item())
        losses_mmd.update(L_mmd.item())
        losses_kd.update(L_kd.item())
        labeled_acc.update(acc_l.item())
        unlabeled_acc.update(acc_ul.item())
        batch_time.update(time.time() - end)
        masks_ssl.update(mask.mean())
        masks_mmd.update(mmd_mask_u.mean())
        masks_kd.update(kd_mask.mean())
        end = time.time()
        
        if (batch_idx+1) % 10==0:
            bar.suffix = ("{batch:4}/{iter:4}. LR:{lr:.6f}. Data:{dt:.3f}s. Batch:{bt:.3f}s. Loss:{loss:.4f}. Loss_CE:{loss_ce:.4f}. Loss_SSL:{loss_ssl:.4f}. Loss_MMD:{loss_mmd:.4f}. Loss_KD:{loss_kd:.4f}. Acc_L:{acc_l:.4f}.  Acc_U:{acc_u:.4f}. Loss_CE:{loss_ce:.4f}. Mask_SSL:{m_ssl:.4f}. Mask_MMD:{m_mmd:.4f}. Mask_KD:{m_kd:.4f}.".format(
                    batch=batch_idx+1,
                    iter=n_iter,
                    lr=cur_lr,
                    dt=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_ce=losses_ce.avg,
                    loss_ssl=losses_ssl.avg,
                    loss_mmd=losses_mmd.avg,
                    loss_kd=losses_kd.avg,
                    acc_l=labeled_acc.avg,
                    acc_u=unlabeled_acc.avg,
                    m_ssl=masks_ssl.avg,m_mmd=masks_mmd.avg, m_kd=masks_kd.avg))
            bar.next()
    
    bar.finish()
    return (losses.avg, losses_ce.avg, losses_ssl.avg, losses_mmd.avg, losses_kd.avg,labeled_acc.avg, unlabeled_acc.avg, masks_ssl.avg, masks_mmd.avg,masks_kd.avg)


def evaluate(raw_model, eval_model, loader, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    raw_acc = AverageMeter()
    end = time.time()
    # switch to evaluate mode
    if eval_model is None:
        eval_model = raw_model
    raw_model.eval()
    eval_model.eval()
    full_targets, full_outputs=None, None
    with torch.no_grad():
        bar = Bar('Evaluating', max=len(loader))
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = eval_model(inputs)
            raw_outputs = raw_model(inputs)
            if full_outputs is None:
                full_targets = targets
                full_outputs = outputs
            else:
                full_targets = torch.cat((full_targets,targets))
                full_outputs = torch.cat((full_outputs,outputs))
            # measure accuracy and record loss
            loss = F.cross_entropy(outputs, targets)
            prec1 = (outputs.max(1)[1] == targets).float().mean()
            raw_prec1 = (raw_outputs.max(1)[1] == targets).float().mean()
            losses.update(loss.item(), inputs.shape[0])
            acc.update(prec1.item(), inputs.shape[0])
            raw_acc.update(raw_prec1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if (batch_idx+1)%10==0:
                bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Loss: {loss:.4f} | Acc: {acc: .4f} | Raw-Acc.: {racc: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=acc.avg,
                        racc=raw_acc.avg,
                        )
                bar.next()
        bar.finish()
    raw_met = raw_acc.avg
    raw_model.train()
    eval_model.train()
    return (losses.avg, acc.avg, raw_met)
