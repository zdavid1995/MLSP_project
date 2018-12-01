import torch
import torch.nn as nn
import numpy as np

def make_features(in_channels=3):
	"""
	Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
	https://arxiv.org/pdf/1412.6806.pdf
	Use a AvgPool2d to pool and then your Flatten layer as your final layers.
	You should have a total of exactly 23 layers of types:
	- nn.Dropout
	- nn.Conv2d
	- nn.ReLU
	- nn.AvgPool2d
	- Flatten
	:return: a nn.Sequential model
	"""
	return nn.Sequential(
		nn.Dropout(p=0.2),
		nn.Conv2d(in_channels=in_channels,out_channels=96,kernel_size=3,stride=1,padding=1),
		nn.ReLU(),
		nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1),
		nn.ReLU(),
		nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=2,padding=1),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		nn.Conv2d(in_channels=96,out_channels=192,kernel_size=3,stride=1,padding=1),
		nn.ReLU(),
		nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
		nn.ReLU(),
		nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=2,padding=1),
		nn.ReLU(),
		nn.Dropout(p=0.5),
		nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=0),
		nn.ReLU(),
		nn.Conv2d(in_channels=192,out_channels=192,kernel_size=1,stride=1,padding=0),
		nn.ReLU(),
		nn.Conv2d(in_channels=192,out_channels=256,kernel_size=1,stride=1,padding=0),
		nn.ReLU(),
		nn.AdaptiveAvgPool2d((1,1))
		)

class All_CNN(nn.Module):
	def __init__(self,im_grayscale=True,im_gradients=True):
		super(All_CNN,self).__init__()
		# Hard coded block that computes the image gradients from grayscale.
		# Not learnt.
		self.im_gradients = im_gradients
		self.n_input_channels = 1 if im_grayscale else 3
		conv_gradients = nn.Conv2d(self.n_input_channels, 2, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = make_features(2 if self.im_gradients else self.n_input_channels)

		if self.im_gradients:
			self.pre_processing = nn.Sequential(conv_gradients)
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
	def forward(self,x):
		if self.im_gradients:
			x = self.pre_processing(x)
		x = self.features(x)
		x = x.view(x.size(0),-1)
		return x

class Decoder(nn.Module):

	def __init__(self,input_shape=256,nhid=192,num_classes=10):
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
