from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1)  # (x_size, 1, dim)
	y = y.unsqueeze(0)  # (1, y_size, dim)
	tiled_x = x.expand(x_size, y_size, dim)
	tiled_y = y.expand(x_size, y_size, dim)
	kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
	return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd_loss(x, y,reduction=None):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
	return mmd


def reg_MMD(fm_labeled, fm_unlabeled):
	fml = F.adaptive_avg_pool2d(fm_labeled, (1, 1))
	fmu = F.adaptive_avg_pool2d(fm_unlabeled, (1, 1))
	fml = fml.reshape((fml.shape[0], -1))
	fmu = fmu.reshape((fmu.shape[0], -1))
	fea_loss = mmd_loss(fml, fmu)
	# print(torch.norm(fml-fmu), fea_loss)
	return fea_loss


def reg_KD(fm_src, fm_tgt,):
	b, c, h, w = fm_src.shape
	fea_loss = torch.norm(fm_src - fm_tgt) / (h * c )
	return fea_loss


def softmax_mse_loss(input_logits, target_logits,reduction='mean'):
	"""Takes softmax on both sides and returns MSE loss

	Note:
	- Returns the sum over all examples. Divide by the batch size afterwards
	  if you want the mean.
	- Sends gradients to inputs but not the targets.
	"""
	input_softmax = F.softmax(input_logits, dim=1)
	target_softmax = F.softmax(target_logits, dim=1)
	#num_classes = input_logits.size()[1]
	return F.mse_loss(input_softmax, target_softmax, reduction=reduction) #/ num_classes


def softmax_kl_loss(input_logits, target_logits,reduction='mean'):
	"""Takes softmax on both sides and returns KL divergence

	Note:
	- Returns the sum over all examples. Divide by the batch size afterwards
	  if you want the mean.
	- Sends gradients to inputs but not the targets.
	"""
	input_log_softmax = F.log_softmax(input_logits, dim=1)
	target_softmax = F.softmax(target_logits, dim=1)
	return F.kl_div(input_log_softmax, target_softmax, reduction=reduction)
    
class Distribution_Loss(nn.Module):
	def __init__(self, loss='softmax_mse', reduction='mean'):#['mse','kl','softmax_mse','softmax_kl']
		super(Distribution_Loss, self).__init__()
		self.check_shape = True
		loss=loss.lower()
		if loss=='mse':
			criterion = F.mse_loss
		elif loss=='softmax_mse':
			criterion = softmax_mse_loss
		elif loss=='kl':
			criterion = F.kl_div
		elif loss=='softmax_kl':
			criterion = softmax_kl_loss
		elif loss=='mmd':
			criterion = mmd_loss
			self.check_shape = False
		else:
			raise NotImplementedError
		self.loss_name = loss
		self.criterion = criterion
		self.reduction = reduction

	def forward(self, input_logits, target_logits,mask=None, reduction=None):
		if self.check_shape:
			assert input_logits.size() == target_logits.size()
		if reduction is None:
			reduction=self.reduction

		input_logits = F.adaptive_avg_pool2d(input_logits, (1, 1))
		target_logits = F.adaptive_avg_pool2d(target_logits, (1, 1))
		input_logits = input_logits.reshape((input_logits.shape[0], -1))
		target_logits = target_logits.reshape((target_logits.shape[0], -1))
		loss = self.criterion(input_logits, target_logits,reduction=reduction)
		if 'softmax' not in self.loss_name and 'mmd' not in self.loss_name:
			loss = loss/10000
		if len(loss.shape)>1:
			loss=loss.sum(1)
			if mask is not None:
				loss = (loss*mask).sum()/(mask.sum() if mask.sum()>0 else 1)
			else:
				loss = loss.mean()
		#loss=loss/2.5
		return loss
