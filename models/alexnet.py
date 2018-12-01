#!/usr/bin/python
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class AlexNetFeatureNet(nn.Module):

	def __init__(self,
				 im_grayscale: bool=True,
				 im_gradients: bool=True) -> None:
		"""
		Build AlexNet Architecture (Encoder part only). as in Noise as Target paper implementation.
		:param im_grayscale: whether to use grayscale or RGB input image.
		:param im_gradients: whether to use a fixed sobel operator at the start of the network instead of the raw pixels.
		"""
		super().__init__()

		self.im_gradients = im_gradients
		self.n_input_channels = 1 if im_grayscale else 3

		self.features = nn.Sequential(
			# nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			# nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			# nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.Conv2d(192, 384, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			# nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.init_weights()

		conv_gradients = nn.Conv2d(self.n_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)
		if self.im_gradients:
			self.pre_processing = nn.Sequential(conv_gradients)
			dx = np.array([[[-1.0, 0.0, 1.0],
							[-2.0, 0.0, 2.0],
							[-1.0, 0.0, 1.0]]], dtype=np.float32)
			dy = np.array([[[-1.0, -2.0, -1.0],
							[0.0, 0.0, 0.0],
							[1.0, 2.0, 1.0]]], dtype=np.float32)
			n_conv_gradients = torch.from_numpy(
				np.repeat(
					np.concatenate([dx, dy])[:, np.newaxis, :, :],
					self.n_input_channels,
					axis=1
				)
			)
			conv_gradients.weight = nn.Parameter(data=n_conv_gradients, requires_grad=False)

	def init_weights(self):
		for l in self.features:
			if isinstance(l, nn.Conv2d):
				torch.nn.init.xavier_uniform_(l.weight.data)
			elif isinstance(l, nn.BatchNorm2d):
				l.weight.data.fill_(1)
				l.bias.data.zero_()

	def forward(self, x):
		if self.im_gradients:
			x = self.pre_processing(x)
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		return x

	def test_forward(self, input_shape):
		return self.forward(torch.randn([1] + list(input_shape))).shape[1:]


class Decoder(nn.Module):

	def __init__(self,input_shape=256,nhid=512,num_classes=256):
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
