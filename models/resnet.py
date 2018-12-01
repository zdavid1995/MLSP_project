#!/usr/bin/python
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np



class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
	def __init__(self,block,layers,im_grayscale=True,im_gradients=True):
		"""
		Initialize the model

		:param block: the class used to instantiate a resnet block
		:param layers: the number of layers per block
		:param im_grayscale: whether the input image is grayscale or RGB.
		:param im_gradients: whether to use a fixed sobel operator at the start of the network instead of the raw pixels.
		"""
		super(ResNet,self).__init__()

		self.inplanes = 64
		self.im_gradients = im_gradients
		self.n_input_channels = 1 if im_grayscale else 3

		# Hard coded block that computes the image gradients from grayscale.
		# Not learnt.
		conv_gradients = nn.Conv2d(self.n_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)

		if self.im_gradients:
			self.pre_processing = nn.Sequential(conv_gradients)

		self.features = []
		self.features.append(nn.Conv2d(2 if self.im_gradients else self.n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
		self.features.append(nn.BatchNorm2d(64))
		self.features.append(nn.ReLU(inplace=True))
		self.features.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
		self.features.append(self._make_layer(block, 64, layers[0]))
		self.features.append(self._make_layer(block, 128, layers[1], stride=2))
		self.features.append(self._make_layer(block, 256, layers[2], stride=2))
		self.features.append(self._make_layer(block, 512, layers[3], stride=2))
		self.features.append(nn.AdaptiveAvgPool2d((1,1)))
		self.features= nn.Sequential(*self.features)



		# self.conv1 = nn.Conv2d(2 if self.im_gradients else self.n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		# self.bn1 = nn.BatchNorm2d(64)
		# self.relu = nn.ReLU(inplace=True)
		# self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		# self.layer1 = self._make_layer(block, 64, layers[0])
		# self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		# self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		# self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		# self.avgpool = nn.AdaptiveAvgPool2d((1,1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


		if self.im_gradients:
			# override weights for preprocessing part.
			dx = np.array([[[-1.0, 0.0, 1.0],
							[-2.0, 0.0, 2.0],
							[-1.0, 0.0, 1.0]]], dtype=np.float32)
			dy = np.array([[[-1.0, -2.0, -1.0],
							[0.0, 0.0, 0.0],
							[1.0, 2.0, 1.0]]], dtype=np.float32)
			_conv_grad = torch.from_numpy(
				np.repeat(
					np.concatenate([dx, dy])[:, np.newaxis, :, :],
					self.n_input_channels,
					axis=1
				)
			)
			conv_gradients.weight = nn.Parameter(data=_conv_grad, requires_grad=False)

		

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		if self.im_gradients:
			x = self.pre_processing(x)
		x = self.features(x)
		# x = self.conv1(x)
		# x = self.bn1(x)
		# x = self.relu(x)
		# x = self.maxpool(x)

		# x = self.layer1(x)
		# x = self.layer2(x)
		# x = self.layer3(x)
		# x = self.layer4(x)

		# x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return x

	def testing_forward(self, input_shape):
		return self.forward(torch.randn([1] + list(input_shape))).shape[1:]


class Decoder(nn.Module):

	def __init__(self,input_shape=512,nhid=256,num_classes=257):
		super().__init__()
		self.classifier = []
		self.classifier.append(nn.Linear(input_shape, nhid))
		self.classifier.append(nn.BatchNorm1d(nhid))
		self.classifier.append(nn.ReLU())
		self.classifier.append(nn.Linear(nhid,nhid))
		self.classifier.append(nn.BatchNorm1d(nhid))
		self.classifier.append(nn.ReLU())
		self.classifier.append(nn.Linear(nhid,num_classes))
		self.classifier = nn.Sequential(*self.classifier)
		self.init_weights()

	def init_weights(self):
		for l in self.classifier:
			if isinstance(l, nn.Linear):
				torch.nn.init.xavier_uniform_(l.weight.data)
			elif isinstance(l, nn.BatchNorm1d):
				l.weight.data.fill_(1)
				l.bias.data.zero_()

	def forward(self, x):
		return self.classifier(x)
