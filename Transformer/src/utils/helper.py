import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import math
import copy
import time
from torch.autograd import Variable
from src.utils.bleu import compute_bleu
import pdb

class Syn_Voc:
	def __init__(self):
		self.frequented = False
		# self.w2id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '_': 10, '</s>': 11, '<s>': 12, 'PAD': 13}
		# self.id2w = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '_', 11: '</s>', 12: '<s>', 13: 'PAD'}
		# self.w2c = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '_': 0, '</s>': 0, '<s>': 0, 'PAD': 0}
		self.w2id = {'<s>': 0, '</s>': 1, 'unk': 2, 'PAD': 3}
		self.id2w = {0: '<s>', 1: '</s>', 2: 'unk', 3: 'PAD'}
		self.w2c = {}
		self.nwords = 4
		# self.nwords = 14

	def add_word(self, word):
		if word not in self.w2id: # IT SHOULD NEVER GO HERE!!
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			self.w2c[word] += 1

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, args, train_dataloader):
		for data in train_dataloader:
			for sent in data['src']:
				self.add_sent(sent)
			for sent in data['trg']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords

	def add_to_vocab_dict(self, args, dataloader):
		for data in dataloader:
			for sent in data['src']:
				self.add_sent(sent)
			for sent in data['trg']:
				self.add_sent(sent)

		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords


def stack_to_string(stack):
	op = ""
	for i in stack:
		if op == "":
			op = op + i
		else:
			op = op + ' ' + i
	return op
 
def bleu_scorer(ref, hyp, script='default'):
	'''
		Bleu Scorer (Send list of list of references, and a list of hypothesis)
	'''
	refsend = []
	for i in range(len(ref)):
		refsi = []
		for j in range(len(ref[i])):
			refsi.append(ref[i][j].split())
		refsend.append(refsi)

	gensend = []
	for i in range(len(hyp)):
		gensend.append(hyp[i].split())

	if script == 'nltk':
		 metrics = corpus_bleu(refsend, gensend)
		 return [metrics]

	metrics = compute_bleu(refsend, gensend)
	return metrics

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)

""" Noam Optimizer """

class NoamOpt:
	"Optim wrapper that implements rate."

	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()

	def rate(self, step=None):
		"Implement `lrate` above"
		if step is None:
			step = self._step
		return self.factor * \
			(self.model_size ** (-0.5) *
			 min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
	return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


""" Label Smoothing """


class LabelSmoothing(nn.Module):
	"Implement label smoothing."

	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None

	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.nelement() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist
		return self.criterion(x, Variable(true_dist, requires_grad=False))


''' Gpu initialization '''


def gpu_init_pytorch(gpu_num):
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
	return device



def write_meta(args, fh):
	layers = args.layers
	heads= args.heads
	d_model = args.d_model
	d_ff = args.d_ff
	max_len = args.max_length
	dropout = args.dropout
	BATCH_SIZE = args.batch_size
	epochs= args.epochs

	fh.write('Layers: {}\n, Heads: {}\n, d_model: {}\n, d_model: {}\n, d_ff: {}\n, dropout {}\n, batch_size: {}\n, epochs: {}\n\n'.format(layers,heads,d_model,d_ff,max_len,dropout,BATCH_SIZE,epochs))

	return


''' Loss Computation '''


class LossCompute:

	def __init__(self, generator, criterion, device, opt=None):
		self.device = device
		self.generator = generator
		self.criterion = criterion
		self.opt = opt

	def __call__(self, out, targets, normalize):
		generator = self.generator.to(self.device)
		final_out = generator(out)
		loss = self.criterion(final_out.contiguous().view(-1, final_out.size(-1)), targets.contiguous().view(-1)) / normalize.float().item()
		loss.backward()

		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()

		return loss.item() * normalize.float().item()

def create_save_directories(path):
	if not os.path.exists(path):
		os.makedirs(path)


def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}.pt'.format(ckpt))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)


def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path: including the run_name
			logger variable: to log messages
		Returns:
			checkpoint: path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		ckpt_path = ckpts[0]
		logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path

def load_checkpoint(model, mode, ckpt_path, logger, device):
	start_epoch = None
	train_loss = None
	val_loss = None
	voc1 = None
	voc2 = None
	score = -1

	checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	start_epoch = checkpoint['epoch']
	train_loss  =checkpoint['train_loss']
	val_loss = checkpoint['val_loss']
	voc1 = checkpoint['voc1']
	voc2 = checkpoint['voc2']
	score = checkpoint['val_acc_score']

	model.to(device)

	if mode == 'train':
		model.train()
	else:
		model.eval()

	logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

	return start_epoch, train_loss, val_loss, score, voc1, voc2