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


cuda = torch.cuda.is_available()
weight_decay_rate = 1e-5
learning_rate = 1e-3
batch_size = 128
im_grayscale = True
im_gradient = True
update_targets_period = 1
train_decoder_period = 10
emb_dim = 256
epochs = 100
num_classes = 10
use_cosine = True
triplet = True


class CosineLoss(nn.Module):
	def __init__(self):
		super(CosineLoss,self).__init__()
		self.loss_fn = nn.CosineSimilarity()
	def forward(self,embeddings,targets):
		return 1 - self.loss_fn(embeddings,targets)

def triplet_loss(embeddings,targets,criterion,margin=0.5):
	negative_distance = 0
	for i,emb in enumerate(embeddings):
		t_loss = criterion(emb.unsqueeze(0).repeat((len(targets)),1),targets)
		t_loss[i] = 0
		negative_distance += t_loss.sum()/len(targets)
		negative_distance += t_loss.sum()/len(targets)
	positive_distance = criterion(embeddings,targets).sum()/len(targets)
	return positive_distance - negative_distance + margin




def create_targets(n,emb_dim):
	targets = np.random.normal(0,1,(n,emb_dim))
	t_norms = np.expand_dims(np.linalg.norm(targets,axis=1),1)
	return targets/t_norms


def image_transforms():
	im_transforms = []
	if im_grayscale:
		im_transforms.append(transforms.Grayscale())
	im_transforms.append(transforms.Resize((224,224)))
	im_transforms.append(transforms.Lambda(lambda img: img.convert('L')))
	im_transforms.append(transforms.RandomHorizontalFlip())
	im_transforms.append(transforms.ToTensor())
	return transforms.Compose(im_transforms)


transform_obj = image_transforms()
encoder = AlexNetFeatureNet(im_grayscale=im_grayscale,im_gradients=im_gradient)
decoder = Decoder(num_classes=num_classes)
if not use_cosine:
	encoder_loss_fn = nn.MSELoss(reduce=False)
else:
	encoder_loss_fn = CosineLoss()
# encoder_closs_fn = nn.CosineSimilarity()
decoder_loss_fn = nn.CrossEntropyLoss()


if cuda:
	encoder.cuda()
	decoder.cuda()
	decoder_loss_fn = decoder_loss_fn.cuda()
	encoder_loss_fn = encoder_loss_fn.cuda()


encoder_optim = torch.optim.Adam(encoder.features.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_optim, milestones=[10, 20], gamma=0.5)


train_loader = torchvision.datasets.CIFAR10(os.getcwd(),download=True,transform=image_transforms())
train_data = train_loader.train_data.transpose((0,3,1,2))
train_data = np.expand_dims(train_data.sum(axis=1)/3,1)
print(train_data.shape)
train_labels = np.array(train_loader.train_labels)
train_len = len(train_data)
train_targets = create_targets(train_len,emb_dim)

test_loader = torchvision.datasets.CIFAR10(os.getcwd(),download=True,train=False,transform=image_transforms())
test_data = test_loader.test_data.transpose((0,3,1,2))
test_data = np.expand_dims(test_data.sum(axis=1)/3,1)
print(test_data.shape)
test_labels = np.array(test_loader.test_labels)
test_len = len(test_data)

writer = SummaryWriter(log_dir="./logs")



def calc_optimal_target_permutation(feats, targets):
	"""
	Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.

	:param feats: the learnt features (given some input images)
	:param targets: the currently assigned targets.
	:return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
	"""
	# Compute cost matrix
	cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
	# calc SSE between all features and targets
	for i in range(feats.shape[0]):
		cost_matrix[:, i] = np.sum(np.square(feats-targets[i, :]), axis=1)
	# Permute the targets based on hungarian algorithm optimisation
	_, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
	targets[range(feats.shape[0])] = targets[col_ind]
	return targets

def train():
	enc_losses = []
	dec_losses = []
	dec_accs = []
	best_accuracy = 0
	steps = 0
	for epoch in range(epochs):
		scheduler.step(epoch)
		update_targets = bool(((epoch+1) % update_targets_period) == 0)
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
			targets = train_targets[r_idx[i:i + bsz]]
			Y = Variable(torch.from_numpy(train_targets[r_idx[i:i+bsz]]))

			encoder_optim.zero_grad()
			outputs = encoder(X)


			if update_targets:
				t_feats = outputs.cpu().data.numpy()
				train_targets[r_idx[i:i+bsz]] = calc_optimal_target_permutation(t_feats,train_targets[r_idx[i:i+bsz]])

			targets = torch.FloatTensor(train_targets[r_idx[i:i+bsz]])
			if cuda:
				targets = targets.cuda()
			# train encoder
			if triplet:
				encoder_loss = triplet_loss(outputs,targets,encoder_loss_fn)
			else:
				encoder_loss = encoder_loss_fn(outputs, targets).sum()/bsz
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
				logits = decoder(outputs)
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
		prepend = "3let_" if triplet else ""
		if not use_cosine:
			np.save(prepend + "decoder_accuracies_L2.npy",np.array(dec_accs))
			np.save(prepend + "decoder_losses_L2.npy",np.array(dec_losses))
			np.save(prepend + "encoder_losses_L2.npy",np.array(enc_losses))
		else:
			np.save(prepend + "decoder_accuracies_cosine.npy",np.array(dec_accs))
			np.save(prepend + "decoder_losses_cosine.npy",np.array(dec_losses))
			np.save(prepend + "encoder_losses_cosine.npy",np.array(enc_losses))
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
