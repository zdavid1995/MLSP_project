import os
import argparse
import scipy
import scipy.optimize
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets
from models.alexnet import *
from models.ASoftMax import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


cuda = torch.cuda.is_available()
weight_decay_rate = 1e-5
learning_rate = 1e-3
batch_size = 128
im_grayscale = True
im_gradient = True
train_decoder_period = 1
emb_dim = 256
epochs = 100
num_classes = 10


train_loader = torchvision.datasets.CIFAR10(os.getcwd(),download=True)
train_data = train_loader.train_data.transpose((0,3,1,2))
train_data = np.expand_dims(train_data.sum(axis=1)/3,1)
print(train_data.shape)
train_labels = np.array(train_loader.train_labels)
train_len = len(train_data)
train_targets = np.arange(train_len)
# train_targets = create_targets(train_len,emb_dim)

test_loader = torchvision.datasets.CIFAR10(os.getcwd(),download=True,train=False)
test_data = test_loader.test_data.transpose((0,3,1,2))
test_data = np.expand_dims(test_data.sum(axis=1)/3,1)
print(test_data.shape)
test_labels = np.array(test_loader.test_labels)
test_len = len(test_data)



encoder = AlexNetFeatureNet(im_grayscale=im_grayscale,im_gradients=im_gradient)
as_decoder = AngleLinear(in_features=emb_dim,out_features=train_len)
decoder = Decoder(num_classes=num_classes)

encoder_loss_fn = AngleLoss()
decoder_loss_fn = nn.CrossEntropyLoss()

encoder_optim = torch.optim.Adam(encoder.features.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optim, milestones=[10, 20], gamma=0.5)


if cuda:
	as_decoder.cuda()
	encoder.cuda()
	decoder.cuda()
	decoder_loss_fn = decoder_loss_fn.cuda()
	encoder_loss_fn = encoder_loss_fn.cuda()

writer = SummaryWriter(log_dir="./logs")


def train():
	enc_losses = []
	dec_losses = []
	dec_accs = []
	best_accuracy = 0
	steps = 0
	for epoch in range(epochs):
		scheduler.step(epoch)
		train_decoder = bool(((epoch+1) % train_decoder_period) == 0)
		encoder.train()
		r_idx = np.random.permutation(train_len)
		if train_decoder:
			decoder.train()
		for i in tqdm(range(0,train_len,batch_size)):
			bsz = min(batch_size,train_len - i)
			X = torch.from_numpy(train_data[r_idx[i:i+bsz]]).float()
			if cuda:
				X = X.cuda()
			encoder_optim.zero_grad()
			outputs_enc = encoder(X)
			outputs_as = as_decoder(output_enc)
			targets = torch.LongTensor(train_targets[r_idx[i:i+bsz]])
			if cuda:
				targets = targets.cuda()
			encoder_loss = encoder_loss_fn(output_as, targets)
			encoder_loss.backward(retain_graph=True)
			encoder_optim.step()

			if i % 100 == 0:
				# idx_step = int(((epoch+1)/train_decoder_period)*(i/100))
				writer.add_scalar('encoder_loss', encoder_loss.item(),steps)
				enc_losses.append(encoder_loss.item())
			if train_decoder:
				y = torch.LongTensor(train_labels[r_idx[i:i+bsz]])
				if cuda:
					y = y.cuda()
				decoder_optim.zero_grad()
				logits = decoder(outputs_enc)
				decoder_loss = decoder_loss_fn(logits, y)
				decoder_loss.backward()
				decoder_optim.step()

				if i % 100 == 0:
					# idx_step = int(((epoch+1)/train_decoder_period)*(i/100))
					writer.add_scalar('decoder_loss', decoder_loss.item(),steps)
					dec_losses.append(decoder_loss.item())
			steps += 1
		# set models to eval mode and validate on test set.
		encoder.eval()
		decoder.eval()
		test_loss = 0.0
		n_correct = 0
		for i in tqdm(range(0,test_len,batch_size)):
			decoder_optim.zero_grad()
			bsz = min(batch_size,test_len - i)
			X = torch.FloatTensor(test_data[i:i+bsz])
			Y = torch.LongTensor(test_labels[i:i+bsz])
			if cuda:
				X = X.cuda()
				Y = Y.cuda()
			outputs = encoder(X)
			logits = decoder(outputs)
			loss = decoder_loss_fn(logits, Y)
			test_loss += loss.cpu().detach().item()
			preds = torch.argmax(logits.cpu().data,dim=-1).numpy().astype(int)
			n_correct += sum(preds == test_labels[i:i+bsz])

		accuracy = float(n_correct)/float(test_len)
		writer.add_scalar('accuracy',accuracy,steps)
		dec_accs.append(accuracy)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			print(f'saving best encoder / decoder pair....',accuracy)
			torch.save(encoder.state_dict(),"ENC_STATE.pt")
			torch.save(decoder.state_dict(),"DEC_STATE.pt")


		np.save("decoder_accuracies_angular.npy",np.array(dec_accs))
		np.save("decoder_losses_angular.npy",np.array(dec_losses))
		np.save("encoder_losses_angular.npy",np.array(enc_losses))
			# state = {
			# 		'encoder_state_dict': encoder.state_dict(),
			# 		'decoder_state_dict': decoder.state_dict(),

			# 		'encoder_optim': encoder_optim.state_dict(),
			# 		'decoder_optim': decoder_optim.state_dict(),

			# 		'best_acc': acc_score,
			# 		'epoch': epoch+1,
			# 	}
			# 	if not os.path.isdir(args.checkpoint_dir):
			# 		os.mkdir(args.checkpoint_dir)
			# 	torch.save(state, os.path.join(args.checkpoint_dir, f'chkpt_full_{epoch}.pkl'))


if __name__ == '__main__':
	try:
		train()
	except KeyboardInterrupt:
		# Free up all cuda memory
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
