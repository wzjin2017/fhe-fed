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
torch.manual_seed(50)
for tid in range(10):
	print(torch.__version__, torchvision.__version__)
	interval = 10
	class_num = 100
	protected_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	protected_layers.remove(tid) 
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


	class cnn(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv1 = nn.Conv2d(3, 6, 5)
			self.pool = nn.MaxPool2d(2, 2)
			self.conv2 = nn.Conv2d(6, 16, 5)
			self.fc1 = nn.Linear(16 * 5 * 5, 120)
			self.fc2 = nn.Linear(120, 84)
			self.fc3 = nn.Linear(84, 10)

		def forward(self, x):
			x = self.pool(F.relu(self.conv1(x)))
			x = self.pool(F.relu(self.conv2(x)))
			x = torch.flatten(x, 1) # flatten all dimensions except batch
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x
		
	class CNN_OriginalFedAvg(torch.nn.Module):
		"""The CNN model used in the original FedAvg paper:
		"Communication-Efficient Learning of Deep Networks from Decentralized Data"
		https://arxiv.org/abs/1602.05629.
		The number of parameters when `only_digits=True` is (1,663,370), which matches
		what is reported in the paper.
		When `only_digits=True`, the summary of returned model is
		Model:
		_________________________________________________________________
		Layer (type)                 Output Shape              Param #
		=================================================================
		reshape (Reshape)            (None, 28, 28, 1)         0
		_________________________________________________________________
		conv2d (Conv2D)              (None, 28, 28, 32)        832
		_________________________________________________________________
		max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
		_________________________________________________________________
		conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
		_________________________________________________________________
		max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
		_________________________________________________________________
		flatten (Flatten)            (None, 3136)              0
		_________________________________________________________________
		dense (Dense)                (None, 512)               1606144
		_________________________________________________________________
		dense_1 (Dense)              (None, 10)                5130
		=================================================================
		Total params: 1,663,370
		Trainable params: 1,663,370
		Non-trainable params: 0
		Args:
		only_digits: If True, uses a final layer with 10 outputs, for use with the
			digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
			If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
			EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
		Returns:
		A `torch.nn.Module`.
		"""
		def __init__(self, only_digits=True):
			super(CNN_OriginalFedAvg, self).__init__()
			self.only_digits = only_digits
			self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
			self.max_pooling = nn.MaxPool2d(2, stride=2)
			self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
			self.flatten = nn.Flatten()
			self.linear_1 = nn.Linear(3136, 512)
			self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
			self.relu = nn.ReLU()
			# self.softmax = nn.Softmax(dim=1)
		def forward(self, x):
			x = torch.unsqueeze(x, 1)
			x = self.conv2d_1(x)
			x = self.relu(x)
			x = self.max_pooling(x)
			x = self.conv2d_2(x)
			x = self.relu(x)
			x = self.max_pooling(x)
			x = self.flatten(x)
			x = self.relu(self.linear_1(x))
			x = self.linear_2(x)
			# x = self.softmax(self.linear_2(x))
			return x


	def conv1x1(in_planes, out_planes, stride=1):
		"""1x1 convolution"""
		return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

	def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
		"""3x3 convolution with padding"""
		return nn.Conv2d(
			in_planes,
			out_planes,
			kernel_size=3,
			stride=stride,
			padding=dilation,
			groups=groups,
			bias=False,
			dilation=dilation,
		)

	class ResNet(nn.Module):
		def __init__(
			self,
			block,
			layers,
			num_classes=class_num,
			zero_init_residual=False,
			groups=1,
			width_per_group=64,
			replace_stride_with_dilation=None,
			norm_layer=None,
			KD=False,
		):
			super(ResNet, self).__init__()
			if norm_layer is None:
				norm_layer = nn.BatchNorm2d
			self._norm_layer = norm_layer

			self.inplanes = 16
			self.dilation = 1
			if replace_stride_with_dilation is None:
				# each element in the tuple indicates if we should replace
				# the 2x2 stride with a dilated convolution instead
				replace_stride_with_dilation = [False, False, False]
			if len(replace_stride_with_dilation) != 3:
				raise ValueError(
					"replace_stride_with_dilation should be None "
					"or a 3-element tuple, got {}".format(replace_stride_with_dilation)
				)

			self.groups = groups
			self.base_width = width_per_group
			self.conv1 = nn.Conv2d(
				3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
			)
			self.bn1 = nn.BatchNorm2d(self.inplanes)
			self.relu = nn.ReLU(inplace=True)
			# self.maxpool = nn.MaxPool2d()
			self.layer1 = self._make_layer(block, 16, layers[0])
			self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
			self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			self.fc = nn.Linear(64 * block.expansion, num_classes)
			self.KD = KD
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
			# Zero-initialize the last BN in each residual branch,
			# so that the residual branch starts with zeros, and each residual block behaves like an identity.
			# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
			if zero_init_residual:
				for m in self.modules():
					if isinstance(m, Bottleneck):
						nn.init.constant_(m.bn3.weight, 0)
					elif isinstance(m, BasicBlock):
						nn.init.constant_(m.bn2.weight, 0)

		def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
			norm_layer = self._norm_layer
			downsample = None
			previous_dilation = self.dilation
			if dilate:
				self.dilation *= stride
				stride = 1
			if stride != 1 or self.inplanes != planes * block.expansion:
				downsample = nn.Sequential(
					conv1x1(self.inplanes, planes * block.expansion, stride),
					norm_layer(planes * block.expansion),
				)

			layers = []
			layers.append(
				block(
					self.inplanes,
					planes,
					stride,
					downsample,
					self.groups,
					self.base_width,
					previous_dilation,
					norm_layer,
				)
			)
			self.inplanes = planes * block.expansion
			for _ in range(1, blocks):
				layers.append(
					block(
						self.inplanes,
						planes,
						groups=self.groups,
						base_width=self.base_width,
						dilation=self.dilation,
						norm_layer=norm_layer,
					)
				)

			return nn.Sequential(*layers)

		def forward(self, x):
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)  # B x 16 x 32 x 32
			x = self.layer1(x)  # B x 16 x 32 x 32
			x = self.layer2(x)  # B x 32 x 16 x 16
			x = self.layer3(x)  # B x 64 x 8 x 8

			x = self.avgpool(x)  # B x 64 x 1 x 1
			x_f = x.view(x.size(0), -1)  # B x 64
			x = self.fc(x_f)  # B x num_classes
			if self.KD == True:
				return x_f, x
			else:
				return x

	class BasicBlock(nn.Module):
		expansion = 1

		def __init__(
			self,
			inplanes,
			planes,
			stride=1,
			downsample=None,
			groups=1,
			base_width=64,
			dilation=1,
			norm_layer=None,
		):
			super(BasicBlock, self).__init__()
			if norm_layer is None:
				norm_layer = nn.BatchNorm2d
			if groups != 1 or base_width != 64:
				raise ValueError("BasicBlock only supports groups=1 and base_width=64")
			if dilation > 1:
				raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
			# Both self.conv1 and self.downsample layers downsample the input when stride != 1
			self.conv1 = conv3x3(inplanes, planes, stride)
			self.bn1 = norm_layer(planes)
			self.relu = nn.ReLU(inplace=True)
			self.conv2 = conv3x3(planes, planes)
			self.bn2 = norm_layer(planes)
			self.downsample = downsample
			self.stride = stride

		def forward(self, x):
			identity = x

			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bn2(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out += identity
			out = self.relu(out)

			return out


	class Bottleneck(nn.Module):
		expansion = 4

		def __init__(
			self,
			inplanes,
			planes,
			stride=1,
			downsample=None,
			groups=1,
			base_width=64,
			dilation=1,
			norm_layer=None,
		):
			super(Bottleneck, self).__init__()
			if norm_layer is None:
				norm_layer = nn.BatchNorm2d
			width = int(planes * (base_width / 64.0)) * groups
			# Both self.conv2 and self.downsample layers downsample the input when stride != 1
			self.conv1 = conv1x1(inplanes, width)
			self.bn1 = norm_layer(width)
			self.conv2 = conv3x3(width, width, stride, groups, dilation)
			self.bn2 = norm_layer(width)
			self.conv3 = conv1x1(width, planes * self.expansion)
			self.bn3 = norm_layer(planes * self.expansion)
			self.relu = nn.ReLU(inplace=True)
			self.downsample = downsample
			self.stride = stride

		def forward(self, x):
			identity = x

			out = self.conv1(x)
			out = self.bn1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.bn2(out)
			out = self.relu(out)

			out = self.conv3(out)
			out = self.bn3(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out += identity
			out = self.relu(out)

			return out

	def ResNet18():
		return ResNet(BasicBlock, [2, 2, 2, 2])

	def ResNet34():
		return ResNet(BasicBlock, [3,4,6,3])

	def ResNet50():
		return ResNet(Bottleneck, [3,4,6,3])

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
				nn.Linear(768, 100)
			)
			
		def forward(self, x):
			out = self.body(x)
			out = out.view(out.size(0), -1)
			# print(out.size())
			out = self.fc(out)
			return out





	def label_to_onehot(target, num_classes=class_num):
		target = torch.unsqueeze(target, 1)
		onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
		onehot_target.scatter_(1, target, 1)
		return onehot_target

	def cross_entropy_for_onehot(pred, target):
		return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


	# net= CNN_OriginalFedAvg().to(device)
	# net = cnn().to(device)
	net = LeNet().to(device)
	def weights_init(m):
		if hasattr(m, "weight"):
			m.weight.data.uniform_(-0.5, 0.5)
		if hasattr(m, "bias"):
			m.bias.data.uniform_(-0.5, 0.5)
	net.apply(weights_init)

	# from prettytable import PrettyTable
	# def count_parameters(model):
	#     table = PrettyTable(["Layer Name", "Parameters Listed"])
	#     t_params = 0
	#     for name, parameter in model.named_parameters():
	#         if not parameter.requires_grad: continue
	#         param = parameter.numel()
	#         table.add_row([name, param])
	#         t_params+=param
	#     print(table)
	#     print(f"Sum of trained paramters: {t_params}")
	#     return table


	# pytorch_total_params = sum(p.numel() for p in net.parameters())
	# print(pytorch_total_params)
	# table = count_parameters(net)


	# net = ResNet18().to(device)
	# net = ResNet34().to(device)
	# net = ResNet50().to(device)
	# def init_weights_resnet(m):
	#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
	#         nn.init.uniform_(m.weight, -0.5, 0.5)
	#         if m.bias is not None:
	#             nn.init.uniform_(m.bias, -0.5, 0.5)
	# net.apply(init_weights_resnet)


	criterion = cross_entropy_for_onehot

	######### honest partipant #########
	img_index = 25
	gt_data = tp(dst[img_index][0]).to(device)
	gt_data = gt_data.view(1, *gt_data.size())
	gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
	# print(f"gt_label1={gt_label}")
	gt_label = gt_label.view(1, )
	# print(f"gt_label1={gt_label}")
	gt_onehot_label = label_to_onehot(gt_label, num_classes=class_num)

	plt.imshow(tt(gt_data[0].cpu()))
	plt.title("Ground truth image")
	print("GT label is %d." % gt_label.item(), "\nOnehot label is %d." % torch.argmax(gt_onehot_label, dim=-1).item())

	# compute original gradient 
	out = net(gt_data)
	print(f"out={out.size()}")
	print(f"gt_onehot_label={gt_onehot_label.size()}")
	y = criterion(out, gt_onehot_label)
	dy_dx = torch.autograd.grad(y, net.parameters())



	layer_counter = 0
	gradients = []
	for t in dy_dx:
		if layer_counter not in protected_layers:
			gradients.append(t)
		else:
			new_t = torch.from_numpy(np.zeros(t.size())).float().to(device)
			gradients.append(new_t)
		layer_counter += 1
	# print("Number of Layers: "+str(layer_counter))

	dy_dx = tuple(gradients)

	# share the gradients with other clients
	original_dy_dx = list((_.detach().clone() for _ in dy_dx))

	# generate dummy data and label
	dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
	dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

	plt.imshow(tt(dummy_data[0].cpu()))
	plt.title("Dummy data")
	print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

	optimizer = torch.optim.LBFGS([dummy_data, dummy_label] )

	history = []

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
				for t in dummy_dy_dx:  # tuple
					if layer_counter not in protected_layers:
						gradients2.append(t)
				else:
					new_t = torch.from_numpy(np.zeros(t.size())).float().to(device)
					gradients2.append(new_t)
				layer_counter += 1
				dummy_dy_dx = tuple(gradients2)
	#########################



			for gx, gy in zip(dummy_dy_dx, original_dy_dx): # TODO: fix the variablas here
				grad_diff += ((gx - gy) ** 2).sum()
				grad_count += gx.nelement()
			# grad_diff = grad_diff / grad_count * 1000
			grad_diff.backward()
			
			return grad_diff
		
		optimizer.step(closure)
		if iters % interval == 0: 
			current_loss = closure()
			print(iters, "%.4f" % current_loss.item())
		history.append(tt(dummy_data[0].cpu()))
	print("----------------")
	print(len(history))

	plt.figure(figsize=(12, 8))
	for i in range(30):
		ite_num = i * interval + interval - 1 
		plt.subplot(3, 10, i + 1)
		plt.imshow(history[ite_num] )
		plt.title("iter=%d" % (ite_num+1))
		plt.axis('off')
		# filename = 'test.pdf'
		# plt.savefig(filename)

	filename = str(30*interval) + 'ite_protect' + str(protected_layers) + '_' +'.png'
	plt.savefig(filename)

	print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

	# plt.figure(figsize=(12, 8))
	# for i in range(30):
	#   plt.subplot(3, 10, i + 1)
	#   plt.imshow(history[i * 10])
	#   plt.title("iter=%d" % (i * interval))
	#   plt.axis('off')
	# filename = str(30*interval) + 'ite_protect' + str(protected_layers) + '_' +'.png'
	# plt.savefig(filename)

	#print("Dummy label is %d." % torch.argmax(dummy_label, dim=-1).item())

