import os, numpy, random, time, json
import torch
import torch.nn.functional as F
import torch.optim as optim

from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.models.builder import gen_model
from ssl_lib.datasets.builder import gen_dataloader
from ssl_lib.param_scheduler import scheduler
from ssl_lib.utils import Logger
from ssl_lib.trainer.train import train, evaluate
from ssl_lib.trainer.imprint import imprint

# os.environ["CUDA_DEVICE_ORDER"] = PCI_BUS_ID
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

print('pid:', os.getpid())


def main(cfg):
    # set seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA is NOT available")
        device = "cpu"
    # build data loader
    print("load dataset")
    lt_loader, ult_loader, test_loader, num_classes, img_size = gen_dataloader(cfg.data_root, cfg.dataset, cfg=cfg)
    cfg.num_classes = num_classes
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(cfg.alg, cfg)
    # build student model
    model = gen_model(cfg.model, cfg.depth, cfg.widen_factor, num_classes, cfg.pretrained_weight_path, cfg.pretrained,
                      bn_momentum=cfg.bn_momentum).to(device)
    if cfg.imprint:
        model = imprint(model, lt_loader, num_classes, cfg.num_labels, device)
    if cfg.lambda_kd > 0:
        source_model = gen_model(cfg.model, cfg.depth, cfg.widen_factor, 1000, cfg.pretrained_weight_path, True,
                                 bn_momentum=cfg.bn_momentum).to(device)
        for param_s in source_model.parameters():
            param_s.requires_grad = False  # not update by gradient for eval_net
        source_model.eval()
        source_model = torch.nn.DataParallel(source_model)
    else:
        source_model = None
    # build teacher model
    if cfg.ema_teacher:
        ema_teacher = gen_model(cfg.model, cfg.depth, cfg.widen_factor, num_classes, cfg.pretrained_weight_path,
                                cfg.pretrained, bn_momentum=cfg.bn_momentum).to(device)
        ema_teacher.load_state_dict(model.state_dict())
        for param_t in ema_teacher.parameters():
            param_t.requires_grad = False  # not update by gradient for eval_net
        ema_teacher = torch.nn.DataParallel(ema_teacher)
    else:
        ema_teacher = None
    model = torch.nn.DataParallel(model)
    model.train()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model)

    # build optimizer
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params, 'weight_decay': cfg.weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]

    optimizer = optim.SGD(param_list, lr=cfg.lr, momentum=cfg.momentum, weight_decay=0, nesterov=True)
    # set lr scheduler
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration, num_cycles=cfg.num_cycles)

    # init meter
    start_epoch = 0
    log_names = ['Epoch', 'Learning Rate', 'Train Loss', 'Loss CE', 'Loss SSL', 'Loss MMD', 'Loss KD', 'Labeled Acc',
                 'Unlabeled Acc', 'Mask SSL', 'Mask MMD', 'Mask KD',
                 'Test Loss', 'Test Acc.', 'Test Raw Acc.', 'Time']
    if cfg.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(cfg.resume), 'Error: no checkpoint directory found!'
        cfg.save_path = os.path.dirname(cfg.resume)
        checkpoint = torch.load(cfg.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if cfg.ema_teacher:
            ema_teacher.load_state_dict(checkpoint['ema_model'])
        logger = Logger(os.path.join(cfg.save_path, 'log.txt'), title=cfg.task_name, resume=True)
    else:
        logger = Logger(os.path.join(cfg.save_path, 'log.txt'), title=cfg.task_name)
        logger.set_names(log_names)

    print("training")
    test_acc_list = []
    time_record = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        lr = optimizer.param_groups[0]['lr']

        print('\nEpoch: [%d | %d] LR: %f Epoch Time: %.3f min' % (
        epoch, cfg.epochs, lr, (time.time() - time_record) / 60))
        train_loader = zip(lt_loader, ult_loader)
        train_logs = train(epoch, train_loader, model, source_model, ema_teacher, optimizer, lr_scheduler, ssl_alg, cfg,
                           device)
        dtime = time.time() - time_record

        test_loss, test_acc, test_raw_acc = evaluate(model, ema_teacher, test_loader, device)
        test_acc_list.append(test_acc)
        logger.append((epoch, lr) + train_logs + (test_loss, test_acc, test_raw_acc, dtime))

        if (epoch + 1) % cfg.save_every == 0:
            filepath = os.path.join(cfg.save_path, f'{cfg.net_name}_{epoch + 1}.pth')
            torch.save({'epoch': epoch + 1,
                        'model': model.module.state_dict(),
                        'ema_model': ema_teacher.module.state_dict() if cfg.ema_teacher else None,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()}, filepath)

        time_record = time.time()

    accuracies = {}
    for i in [1, 10, 20, 50]:
        print(f"mean test acc. over last {i} checkpoints: {numpy.mean(test_acc_list[-i:])}")
        print(f"median test acc. over last {i} checkpoints: {numpy.median(test_acc_list[-i:])}")
        accuracies[f"mean_last{i}"] = numpy.mean(test_acc_list[-i:])
        accuracies[f"mid_last{i}"] = numpy.median(test_acc_list[-i:])
    with open(os.path.join(cfg.save_path, "test_results.json"), "w") as f:
        json.dump(accuracies, f, sort_keys=True)


if __name__ == "__main__":
    from parser import get_args

    args = get_args()
    print('args:', args)
    main(args)
