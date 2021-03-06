import torch
import torchvision.transforms as tt

from .augmentation_pool import RandAugment, GaussianNoise, GCN, ZCA



class StrongAugmentation:
    """
    Strong augmentation class
    including RandAugment and Cutout
    """
    def __init__(
        self,
        img_size: int,
        mean: list,
        scale: list,
        flip: bool,
        crop: bool,
        alg: str = "fixmatch",
        zca: bool = False,
        cutout_size: float = 0.5,
    ):
        #augmentations = [tt.ToPILImage()]
        augmentations = []
        if flip:
            augmentations += [tt.RandomHorizontalFlip(p=0.5)]
        if crop:
            if img_size==224:
                augmentations += [tt.Resize(int(img_size/0.875)), tt.RandomCrop(img_size)]
            else:
                augmentations += [tt.RandomCrop(img_size, padding=int(img_size*0.125), padding_mode="reflect")]

        augmentations += [
            RandAugment(alg=alg,cutout_size=cutout_size),
            tt.ToTensor(),
        ]
        if zca:
            augmentations += [GCN(), ZCA(mean, scale)]
        else:
            augmentations += [tt.Normalize(mean, scale, True)]
        self.augmentations = tt.Compose(augmentations)

    def __call__(self, img):
        return self.augmentations(img)

    def __repr__(self):
        return repr(self.augmentations)


class WeakAugmentation:
    """
    Weak augmentation class
    including horizontal flip, random crop, and gaussian noise
    """
    def __init__(
        self,
        img_size: int,
        mean: list,
        scale: list,
        flip=True,
        crop=True,
        noise=True,
        zca=False
    ):
        #augmentations = [tt.ToPILImage()]
        augmentations = []
        if flip:
            augmentations.append(tt.RandomHorizontalFlip())
        if crop:
            if img_size==224:
                augmentations += [tt.Resize(int(img_size/0.875)), tt.RandomCrop(img_size)]
            else:
                augmentations += [tt.RandomCrop(img_size, int(img_size*0.125), padding_mode="reflect")]
        augmentations += [tt.ToTensor()]
        if zca:
            augmentations += [GCN(), ZCA(mean, scale)]
        else:
            augmentations += [tt.Normalize(mean, scale, True)]
        if noise:
            augmentations.append(GaussianNoise())
        self.augmentations = tt.Compose(augmentations)

    def __call__(self, img):
        return self.augmentations(img)

    def __repr__(self):
        return repr(self.augmentations)
