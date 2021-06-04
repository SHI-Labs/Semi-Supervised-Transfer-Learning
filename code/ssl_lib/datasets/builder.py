import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from . import utils
from . import dataset_class
from .sampler import RandomSampler, BatchSampler
from ..augmentation.builder import gen_strong_augmentation, gen_weak_augmentation

def _permutation(data, indics):
    new_data={}
    for k in ['images','labels']:
        val = data[k]
        if isinstance(val,np.ndarray):
            new_val = val[indics]
        else:
            new_val=[]
            for ind in indics:
                new_val.append(val[ind])
        new_data[k]=new_val
    return new_data


def __labeled_unlabeled_split(cfg, train_data, test_data, num_classes,):
    np.random.seed(cfg.seed)

    permutation = np.random.permutation(len(train_data["images"]))
    train_data = _permutation(train_data, permutation)
    #train_data["images"] = train_data["images"][permutation]
    #train_data["labels"] = train_data["labels"][permutation]

    l_train_data, ul_train_data = utils.dataset_split(train_data, cfg.num_labels, num_classes)

    return l_train_data, ul_train_data


def gen_dataloader(root, dataset, cfg):
    """
    generate train, val, and test dataloaders

    Parameters
    --------
    root: str
        root directory
    dataset: str
        dataset name, ['cifar10', 'cifar100', 'svhn', 'stl10']
    cfg: argparse.Namespace or something
    logger: logging.Logger
    """
    if dataset == "cub200":
        train_data, test_data = utils.get_cub200(root)
        num_classes = 200
        img_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == "indoor":
        train_data, test_data = utils.get_indoor(root)
        num_classes = 67
        img_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == "cifar10":
        train_data, test_data = utils.get_cifar10(root)
        num_classes = 10
        img_size = 32
        mean = (0.481, 0.457, 0.408)
        std = (0.260, 0.253, 0.268)        
    else:
        raise NotImplementedError

    l_train_data, ul_train_data = __labeled_unlabeled_split(
            cfg, train_data, test_data, num_classes)

    #ul_train_data["images"] = np.concatenate([ul_train_data["images"], l_train_data["images"]], 0)
    #ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], l_train_data["labels"]], 0)
    if cfg.num_unlabels>0:
        np.random.seed(cfg.seed)
        permutation = np.random.permutation(len(ul_train_data["images"]))
        ul_train_data = _permutation(ul_train_data, permutation)
        ul_train_data, _ = utils.dataset_split(ul_train_data, cfg.num_unlabels, num_classes,random=True)
        if isinstance(ul_train_data["images"][0],np.ndarray):
            ul_train_data["images"] = np.concatenate([ul_train_data["images"], l_train_data["images"]], 0)
        else:
            ul_train_data["images"] = ul_train_data["images"] + l_train_data["images"]
        ul_train_data["labels"] = np.concatenate([ul_train_data["labels"], l_train_data["labels"]], 0)
    else:
        ul_train_data = train_data

    #！！！！！！ Following ul_train_dat=train_data may in prev supervised with lambda_kd>0 lambda_mmd>0  
    #if cfg.default_algo_args =='supervised':
    #    ul_train_data = l_train_data
        
    print(f'number of :\n training data: {len(train_data["images"])}\n labeled data: {len(l_train_data["images"])}\n unlabeled data: {len(ul_train_data["images"])}\n test data: {len(test_data["images"])}')

    labeled_train_data = dataset_class.LabeledDataset(l_train_data)
    unlabeled_train_data = dataset_class.UnlabeledDataset(ul_train_data)

    # set augmentation
    # RA: RandAugment, WA: Weak Augmentation
    flags = [True if b == "t" else False for b in cfg.wa.split(".")]
    if cfg.labeled_aug == "RA":
        labeled_augmentation = gen_strong_augmentation(
            img_size, mean, std, flags[0], flags[1], "fixmatch", False,cfg.cutout_size)
    elif cfg.labeled_aug == "WA":
        labeled_augmentation = gen_weak_augmentation(img_size, mean, std, *flags)
    else:
        raise NotImplementedError

    labeled_train_data.transform = labeled_augmentation

    if cfg.unlabeled_aug == "RA":
        unlabeled_augmentation = gen_strong_augmentation(
            img_size, mean, std, flags[0], flags[1], "fixmatch", False,cfg.cutout_size)
    elif cfg.unlabeled_aug == "WA":
        unlabeled_augmentation = gen_weak_augmentation(img_size, mean, std, *flags)
    else:
        raise NotImplementedError

    unlabeled_train_data.weak_augmentation = unlabeled_augmentation
    
    print("labeled augmentation")
    print(labeled_augmentation)
    print("unlabeled augmentation")
    print(unlabeled_augmentation)
    if cfg.strong_aug:
        strong_augmentation = gen_strong_augmentation(
            img_size, mean, std, flags[0], flags[1], "fixmatch", False,cfg.cutout_size)
        unlabeled_train_data.strong_augmentation = strong_augmentation
        print("strong augmentation")
        print(strong_augmentation)

    test_transform=[]
    if img_size==224:
        test_transform += [transforms.Resize(int(img_size/0.875)), transforms.CenterCrop(img_size)]
    test_transform += [transforms.ToTensor()]
    test_transform += [transforms.Normalize(mean, std, True)]
    test_transform = transforms.Compose(test_transform)
    test_data = dataset_class.LabeledDataset(test_data, test_transform)

    sampler_l = RandomSampler(labeled_train_data, replacement=True, num_samples=cfg.per_epoch_steps * cfg.l_batch_size)
    batch_sampler_l = BatchSampler(sampler_l, batch_size=cfg.l_batch_size, drop_last=True)
    l_train_loader = DataLoader(labeled_train_data, batch_sampler=batch_sampler_l, num_workers=cfg.num_workers, pin_memory=True)

    sampler_u = RandomSampler(unlabeled_train_data, replacement=True, num_samples=cfg.per_epoch_steps * cfg.ul_batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size=cfg.ul_batch_size, drop_last=True)
    ul_train_loader = DataLoader(unlabeled_train_data, batch_sampler=batch_sampler_u, num_workers=cfg.num_workers, pin_memory=True)


    test_loader = DataLoader(
        test_data,
        max(128,cfg.l_batch_size+cfg.ul_batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return (
        l_train_loader,
        ul_train_loader,
        test_loader,
        num_classes,
        img_size
    )
