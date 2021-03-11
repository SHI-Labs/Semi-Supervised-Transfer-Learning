import torch
import torchvision
from PIL import Image
import numpy as np

class LoadedImageFolder(torchvision.datasets.ImageFolder):
	def __init__(self, root, transform=None, target_transform=None):
		super(LoadedImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
		self._load_images()

	def _load_images(self):
		data, targets = [], []
		for (path, target) in self.samples:
			try:
				sample = self.loader(path)
			except:
				print('Err:', path)
				continue
			data.append(sample)
			targets.append(target)
		self.data = data
		self.targets = np.array(targets)


	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
	
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]
	
		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
	
		if self.transform is not None:
			if isinstance(img, np.ndarray):
				img = Image.fromarray(img)
			img = self.transform(img)
	
		if self.target_transform is not None:
			target = self.target_transform(target)
	
		return img, target


	def __len__(self):
		return len(self.data)



class LabeledDataset:
    """
    For labeled dataset
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        #image = torch.from_numpy(self.dataset["images"][idx]).float()
        #image = image.permute(2, 0, 1).contiguous() / 255.
        #label = int(self.dataset["labels"][idx])
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        if self.transform is not None:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class UnlabeledDataset:
    """
    For unlabeled dataset
    """
    def __init__(self, dataset, weak_augmentation=None, strong_augmentation=None):
        self.dataset = dataset
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation

    def __getitem__(self, idx):
        #image = torch.from_numpy(self.dataset["images"][idx]).float()
        #image = image.permute(2, 0, 1).contiguous() / 255.
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w_aug_image = self.weak_augmentation(image)
        if self.strong_augmentation is not None:
            s_aug_image = self.strong_augmentation(image)
        else:
            s_aug_image = self.weak_augmentation(image)
        return w_aug_image, s_aug_image, label

    def __len__(self):
        return len(self.dataset["images"])

