from .augmentation_class import WeakAugmentation, StrongAugmentation


def gen_strong_augmentation(img_size, mean, std, flip=True, crop=True, alg="fixmatch", zca=False,cutout_size=0.5):
    return StrongAugmentation(img_size, mean, std, flip, crop, alg, zca,cutout_size)


def gen_weak_augmentation(img_size, mean, std, flip=True, crop=True, noise=True, zca=False):
    return WeakAugmentation(img_size, mean, std, flip, crop, noise, zca)
