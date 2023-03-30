import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
# from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
torch.manual_seed(50)

print(torch.__version__, torchvision.__version__)
for tid in range(10):
	interval = 30
	class_num = 100
	# protected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	# protected_layers.remove(tid) 
	protected_layers = [tid] #[0] # protect the first layer
	dst = datasets.CIFAR100("~/.torch", download=True)
	tp = transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(32),
		transforms.ToTensor()
	])
	tt = transforms.ToPILImage()

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:7"
	print("Running on %s" % device)

	def label_to_onehot(target, num_classes=class_num):
		target = torch.unsqueeze(target, 1)
		onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
		onehot_target.scatter_(1, target, 1)
		return onehot_target

	def cross_entropy_for_onehot(pred, target):
		return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

	def weights_init(m):
		if hasattr(m, "weight"):
			m.weight.data.uniform_(-0.5, 0.5)
		if hasattr(m, "bias"):
			m.bias.data.uniform_(-0.5, 0.5)
		
	class LeNet(nn.Module):
		def __init__(self):
			super(LeNet, self).__init__()
			act = nn.Sigmoid
			self.body = nn.Sequential(
				nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
				act(),
				nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
				act(),
				nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
				act(),
				nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
				act(),
			)
			self.fc = nn.Sequential(
				nn.Linear(768, class_num)
			)
			
		def forward(self, x):
			out = self.body(x)
			out = out.view(out.size(0), -1)
			# print(out.size())
			out = self.fc(out)
			return out
		
	net = LeNet().to(device)
		
	net.apply(weights_init)
	criterion = cross_entropy_for_onehot

	######### honest partipant #########
	img_index = 25
	gt_data = tp(dst[img_index][0]).to(device)
	gt_data = gt_data.view(1, *gt_data.size())
	gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
	gt_label = gt_label.view(1, )
	gt_onehot_label = label_to_onehot(gt_label, num_classes=class_num)
	plt.imshow(tt(gt_data[0].cpu()))
	# file_gt = "gt.png"
	# plt.savefig(file_gt)
	plt.title("Ground truth image")
	print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

	# compute original gradient 
	out = net(gt_data)
	y = criterion(out, gt_onehot_label)
	dy_dx = torch.autograd.grad(y, net.parameters())



	# ########### for fHE
	layer_counter = 0
	gradients = []
	for t1 in dy_dx:  # tuple
		if layer_counter not in protected_layers:
			gradients.append(t1)
		else:
			new_t = torch.from_numpy(np.zeros(t1.size())).float().to(device)
			gradients.append(new_t)
		layer_counter += 1

	dy_dx = tuple(gradients)



	########### FOR FHE


	# share the gradients with other clients
	original_dy_dx = list((_.detach().clone() for _ in dy_dx))

	# generate dummy data and label
	dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
	dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

	plt.imshow(tt(dummy_data[0].cpu()))
	plt.title("Dummy data")
	print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

	from ctypes import sizeof
	optimizer = torch.optim.LBFGS([dummy_data, dummy_label] )

	history = []
	losses = []
	for iters in range(30*interval):
		def closure():
			optimizer.zero_grad()

			pred = net(dummy_data) 
			dummy_onehot_label = F.softmax(dummy_label, dim=-1)
			dummy_loss = criterion(pred, dummy_onehot_label) # TODO: fix the gt_label to dummy_label in both code and slides.
			dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
			
			grad_diff = 0
			grad_count = 0


	######################### code for FHE
			if len(protected_layers) > 0:
				layer_counter = 0
				gradients2 = []
				for t2 in dummy_dy_dx:  # tuple
					if layer_counter not in protected_layers:
						gradients2.append(t2)
					else:
						new_t = torch.from_numpy(np.zeros(t2.size())).float().to(device)
						gradients2.append(new_t)
					layer_counter += 1
				dummy_dy_dx = tuple(gradients2)
	#########################



			for gx, gy in zip(dummy_dy_dx, original_dy_dx): # TODO: fix the variablas here
				grad_diff += ((gx - gy) ** 2).sum()
				grad_count += gx.nelement()
			# grad_diff = grad_diff / grad_count * 1000
			grad_diff.backward()
			losses.append(grad_diff.item())
			return grad_diff
		
		optimizer.step(closure)
		if iters % interval == 0: 
			print(f"{iters}, {losses[iters]}")
		history.append(tt(dummy_data[0].cpu()))

	plt.figure(figsize=(12, 8))
	for i in range(30):
		ite_num = i * interval + interval - 1 
		plt.subplot(3, 10, i + 1)
		plt.imshow(history[ite_num] )
		plt.title("iter=%d" % (ite_num+1))
		plt.axis('off')


	filename = 'CIFAR'+str(class_num)+'_'+ str(30*interval) + 'ite_' + str(protected_layers) + '_' +'.png'
	plt.savefig(filename)


	plt.figure()
	loss=min(losses[0:interval * 30 - 1])
	idx = losses.index(loss)
	print(f"idx = {idx}, len = {len(losses)}")
	plt.imshow(history[idx])
	#plt.title(f"best recovered image: ite={idx}, losses={loss}")
	filename = 'single_' + filename
	plt.savefig(filename)

	print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

