import os
import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision.datasets import CIFAR10
from .dataset_class import LoadedImageFolder


class InfiniteSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        epochs = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(epochs)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)



def get_cub200(root):
    train_root=os.path.join(root, 'CUB_200_2011','train')
    test_root=os.path.join(root, 'CUB_200_2011','test')
    train_data = LoadedImageFolder(train_root)
    test_data = LoadedImageFolder(test_root)
    train_data = {"images": train_data.data,"labels": train_data.targets}
    test_data = {"images": test_data.data,
                 "labels": test_data.targets}
    return train_data, test_data

def get_indoor(root):
    train_root=os.path.join(root, 'indoorCVPR_09','train')
    test_root=os.path.join(root, 'indoorCVPR_09','test')
    train_data = LoadedImageFolder(train_root)
    test_data = LoadedImageFolder(test_root)
    train_data = {"images": train_data.data,"labels": train_data.targets}
    test_data = {"images": test_data.data,
                 "labels": test_data.targets}
    return train_data, test_data

def get_cifar10(root):
    from torchvision.datasets import CIFAR10
    train_data = CIFAR10(root, download=True)
    test_data = CIFAR10(root, False)
    train_data = {"images": train_data.data.astype(np.uint8),
                  "labels": np.asarray(train_data.targets)}
    test_data = {"images": test_data.data.astype(np.uint8), 
                 "labels": np.asarray(test_data.targets)}
    return train_data, test_data

def data_choice(data, indics):
    if isinstance(data,np.ndarray):
        new_data = data[indics]
    else:
        new_data=[]
        for index, flag in enumerate(indics):
            if flag:
                new_data.append(data[index])
    return new_data

def dataset_split(data, num_data, num_classes, random=False):
    """split dataset into two datasets
    
    Parameters
    -----
    data: dict with keys ["images", "labels"]
        each value is numpy.array
    num_data: int
        number of dataset1
    num_classes: int
        number of classes
    random: bool
        if True, dataset1 is randomly sampled from data.
        if False, dataset1 is uniformly sampled from data,
        which means that the dataset1 contains the same number of samples per class.

    Returns
    -----
    dataset1, dataset2: the same dict as data.
        number of data in dataset1 is num_data.
        number of data in dataset1 is len(data) - num_data.
    """
    if num_data==-1:
        dataset1 = {"images": data["images"], "labels": data["labels"]}
        dataset2 = {"images": [], "labels": []}
        return dataset1, dataset2

    dataset1 = {"images": [], "labels": []}
    dataset2 = {"images": [], "labels": []}
    images = data["images"]
    labels = data["labels"]

    # random sampling
    if random:
        dataset1["images"] = images[:num_data]
        dataset1["labels"] = labels[:num_data]
        dataset2["images"] = images[num_data:]
        dataset2["labels"] = labels[num_data:]

    else:
        data_per_class = num_data // num_classes
        for c in range(num_classes):
            c_idx = (labels == c)
            c_imgs = data_choice(images, c_idx)#images[c_idx]
            c_lbls = data_choice(labels, c_idx) #labels[c_idx]
            dataset1["images"].append(c_imgs[:data_per_class])
            dataset1["labels"].append(c_lbls[:data_per_class])
            dataset2["images"].append(c_imgs[data_per_class:])
            dataset2["labels"].append(c_lbls[data_per_class:])
        for k in ("images", "labels"):
            if isinstance(dataset1[k][0],np.ndarray):
                dataset1[k] = np.concatenate(dataset1[k])
                dataset2[k] = np.concatenate(dataset2[k])
            else:
                dat1=[]
                dat2=[]
                for c in range(num_classes):
                    dat1 += dataset1[k][c]
                    dat2 += dataset2[k][c]
                dataset1[k] = dat1
                dataset2[k] = dat2
            

    return dataset1, dataset2


