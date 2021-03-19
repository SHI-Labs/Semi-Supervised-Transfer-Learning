import argparse
import random
from ssl_lib.utils import init_trial_path


def default_parser():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--data_root", "-r", default="data", type=str, help="/path/to/dataset")
    parser.add_argument("--dataset", "-d", default="cub200",  type=str, help="dataset name")
    parser.add_argument("--num_labels", default=400, type=int, help="number of labeled data")
    parser.add_argument("--num_unlabels", default=-1, type=int, help="number of unlabeled data")
    parser.add_argument("--num_workers", default=1, type=int, help="number of thread for CPU parallel")
    # augmentation config
    parser.add_argument("--labeled_aug", default="WA", choices=['WA', 'RA'], type=str, help="type of augmentation for labeled data")
    parser.add_argument("--unlabeled_aug", default="WA", choices=['WA', 'RA'], type=str, help="type of augmentation for unlabeled data")
    parser.add_argument("--wa", default="t.t.f", type=str, help="transformations (flip, crop, noise) for weak augmentation. t and f indicate true and false.")
    parser.add_argument("--strong_aug", default=False,type=bool, help="use strong augmentation (RandAugment) for unlabeled data")
    parser.add_argument("--cutout_size", default=0.5, type=float, help="cutout_size for strong augmentation")
    # optimization config
    parser.add_argument("--model", default="resnet", type=str, help="model architecture") #resner or wideresnetleaky
    parser.add_argument("--depth", default=50,  type=int, help="model depth")
    parser.add_argument("--widen_factor", default=1,  type=int, help="widen factor for wide resnet")
    parser.add_argument("--bn_momentum", default=0.001,  type=float, help="bn momentum for wide resnet")
    parser.add_argument("--l_batch_size", "-l_bs", default=64, type=int, help="mini-batch size of labeled data")
    parser.add_argument("--ul_batch_size", "-ul_bs", default=64, type=int, help="mini-batch size of unlabeled data")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--weight_decay", "-wd", default=0.0001, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd or beta_1 for adam")
    parser.add_argument('--per_epoch_steps', type=int, default=100, help='number of training images for each epoch') #  1000
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs") # iterations 1000000
    parser.add_argument("--warmup_iter", default=0, type=int, help="number of warmup iteration for SSL loss coefficient")
    parser.add_argument("--num_cycles", default=7.9/16, type=float, help="num cycle for CosineAnnealingLR")
    parser.add_argument("--merge_one_batch", default=0 ,type=int, help="interleave of not")
    parser.add_argument("--interleave", default=1 ,type=int, help="interleave of not")
    # SSL common config
    parser.add_argument("--alg", default="cr", choices=['ict', 'cr', 'pl'], type=str,
                        help="ssl algorithm")  # pl: psedu label, fixmatch.
    parser.add_argument("--alpha", default=0.1, type=float, help="parameter for beta distribution in ICT")
    parser.add_argument("--coef", default=0, type=float, help="coefficient for consistency loss")
    parser.add_argument("--ema_teacher", default=False ,type=bool, help="use mean teacher")
    parser.add_argument("--ema_teacher_train", default=False ,type=bool, help="use mean teacher")
    parser.add_argument("--ema_teacher_warmup", default=False ,type=bool, help="warmup for mean teacher")
    parser.add_argument("--ema_teacher_factor", default=0.999, type=float, help="exponential mean avarage factor for mean teacher")
    parser.add_argument("--ema_apply_wd", default=False ,type=bool, help="apply weight decay to ema model")
    parser.add_argument("--threshold", default=None, type=float, help="pseudo label threshold")
    parser.add_argument("--sharpen", default=None, type=float, help="tempereture parameter for sharpening")
    parser.add_argument("--temp_softmax", default=None, type=float, help="tempereture for softmax")
    parser.add_argument("--consistency", "-consis", default="ce", choices=['ce', 'ms'], type=str,
	                    help="consistency type")
    ## AKC and ARC
    parser.add_argument('--reg_warmup', type=int, default=10, )
    parser.add_argument('--reg_warmup_iter', type=int, default=100, ) # 100
    parser.add_argument('--lambda_mmd', type=float, default=50, help='data_mmd_loss_ratio') #50
    parser.add_argument('--mmd_feat_table_l', type=int, default=128, help='feat size for mmd table') # 128
    parser.add_argument('--mmd_feat_table_u', type=int, default=128, help='feat size for mmd table') # 128
    parser.add_argument('--mmd_threshold', default=0.7, type=float, help='kd loss threshold in terms of outputs entropy')
    parser.add_argument('--lambda_kd', type=float, default=0, help='model_kd_loss_ratio') # lambda_kd 1,
    parser.add_argument('--kd_threshold', default=0.7, type=float, help='kd loss threshold in terms of outputs entropy')
    parser.add_argument('--kd_ent_class', type=int, default=1000)
    ## transfer learning
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--imprint', type=int, default=1,help='imprint for pretrained classifier')
    # evaluation checkpoint config
    parser.add_argument("--eval_every", default=1, type=int, help="eval every N epoches")
    parser.add_argument("--save_every", default=200, type=int, help="save every N epoches")
    parser.add_argument("--resume", default=None, type=str, help="path to checkpoint model")
    # misc
    parser.add_argument("--out_dir", default="results", type=str, help="output directory")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--pretrained_weight_path", default="models/ckpt", type=str, help="pretrained_weight_path")

    return parser


def get_args():
    parser = default_parser()
    args = parser.parse_args()
    args.n_imgs_per_epoch = args.per_epoch_steps * args.l_batch_size
    args.iteration = args.epochs * args.per_epoch_steps
    args.net_name = f"{args.model}_{args.depth}_{args.widen_factor}"
    args.task_name = f"{args.dataset}@{args.num_labels}"

    print('default model',args.model,args.depth,args.widen_factor)
    args = init_trial_path(args)
    return args

