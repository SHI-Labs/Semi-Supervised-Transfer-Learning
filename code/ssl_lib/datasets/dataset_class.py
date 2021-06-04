import torch
import torchvision
from PIL import Image
import numpy as np
import os,sys
import pickle

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


class ImageNet32(torchvision.datasets.vision.VisionDataset):
	"""
    CIFAR10 like Imagenet 32*32 dataset

    """
	base_folder = 'imagenet32-batches-py'
	train_list = [
		['train_data_batch_1', '-1'],
		['train_data_batch_2', '-1'],
		['train_data_batch_3', '-1'],
		['train_data_batch_4', '-1'],
		['train_data_batch_5', '-1'],
		['train_data_batch_6', '-1'],
		['train_data_batch_7', '-1'],
		['train_data_batch_8', '-1'],
		['train_data_batch_9', '-1'],
		['train_data_batch_10', '-1'],
	]

	test_list = [
		['val_data', '-1'], ]

	def __init__(self, root, train=True,
	             transform=None, target_transform=None):

		super(ImageNet32, self).__init__(root)
		self.transform = transform
		self.target_transform = target_transform

		self.train = train  # training set or test set

		if self.train:
			downloaded_list = self.train_list
		else:
			downloaded_list = self.test_list

		self.data = []
		self.targets = []

		# now load the picked numpy arrays
		for file_name, checksum in downloaded_list:
			file_path = os.path.join(self.root, file_name)
			if not os.path.exists(file_path):
				file_path = os.path.join(self.root, self.base_folder, file_name)
			with open(file_path, 'rb') as f:
				if sys.version_info[0] == 2:
					entry = pickle.load(f)
				else:
					entry = pickle.load(f, encoding='latin1')
				self.data.append(entry['data'])
				if 'labels' in entry:
					self.targets.extend(entry['labels'])
				else:
					self.targets.extend(entry['fine_labels'])

		self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
		self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC (32,32,3)
		self.targets = np.array(self.targets) - 1

	def cal_mean_std(self):
		self.data_mean = np.mean(self.data, axis=(0, 1, 2))
		self.data_std = np.std(self.data, axis=(0, 1, 2))
		return self.data_mean, self.data_std

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
				img = torchvision.transforms.functional.to_pil_image(img)
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)

	def extra_repr(self):
		return "Split: {}".format("Train" if self.train is True else "Test")



