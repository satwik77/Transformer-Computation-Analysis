import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pdb
import os
import time

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results
from src.utils.sentence_processing import *
from src.utils.evaluate import cal_score, stack_to_string
from src.dataloader import TextDataset
from src.model import *
from src.components.utils import *

# python -m src.main -gpu 3 -batch_size 1500 -max_epochs 40 -ckpt mt.pts

# python -m src.main -gpu 2 -run_name san_cnt100_van3

# python -m src.main -gpu 0 -max_length 6 -batch_size 32 -epochs 20 -dataset count_data -run_name run_counting_task1

try:
	import cPickle as pickle
except ImportError:
	import pickle

import spacy
from torchtext import data, datasets
from src.components.utils import subsequent_mask

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'logs'
model_folder = 'models'
outputs_folder = 'outputs'
result_folder = './out/'
data_path = './data/'
board_path = './runs/'

def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(config, data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug)
		val_set = TextDataset(config, data_path=data_path, dataset=config.dataset,
							  datatype='dev', max_length=config.max_length, is_debug=config.debug)

		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(
			train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(
			val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		logger.info(msg)


		return train_dataloader, val_dataloader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(config, data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))

class Batch:
	"Object for holding a batch of data with mask during training."

	def __init__(self, src, trg=None, pad=0):
		self.src = src
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = \
				self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum()

	@staticmethod
	def make_std_mask(tgt, pad):
		"Create a mask to hide padding and future words."
		tgt_mask = (tgt != pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
	"Keep augmenting batch and calculate total number of tokens + padding."
	global max_src_in_batch, max_tgt_in_batch
	if count == 1:
		max_src_in_batch = 0
		max_tgt_in_batch = 0
	max_src_in_batch = max(max_src_in_batch,  len(new.src))
	max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
	src_elements = count * max_src_in_batch
	tgt_elements = count * max_tgt_in_batch
	return max(src_elements, tgt_elements)

def rebatch(args, device, voc1, voc2, pad_idx, batch):
	"Fix order in torchtext to match ours"
	sent1s = sents_to_idx(voc1, batch['src'], args.max_length)
	sent2s = sents_to_idx(voc2, batch['trg'], args.max_length, flag = 1)
	sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc1, voc2, device, voc1.id2w[pad_idx])
	src, trg = sent1_var.transpose(0, 1), sent2_var.transpose(0, 1)
	return Batch(src, trg, pad_idx)

def main():

	'''Parse Arguments'''
	parser = build_parser()
	args = parser.parse_args()

	'''Specify Seeds for reproducibility'''
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	'''Configs'''
	device = gpu_init_pytorch(args.gpu)

	mode = args.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False
	
	# ckpt= args.ckpt

	run_name = args.run_name
	args.log_path = os.path.join(log_folder, run_name)
	args.model_path = os.path.join(model_folder, run_name)
	args.board_path = os.path.join(board_path, run_name)
	args.outputs_path = os.path.join(outputs_folder, run_name)

	args_file = os.path.join(args.model_path, 'args.p')

	log_file = os.path.join(args.log_path, 'log.txt')

	if args.results:
		args.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(args.dataset))

	logging_var = bool(args.logging)

	if is_train:
		create_save_directories(args.log_path)
		create_save_directories(args.model_path)
		create_save_directories(args.outputs_path)
	else:
		create_save_directories(args.log_path)
		create_save_directories(args.result_path)

	logger = get_logger(run_name, log_file, logging.DEBUG)

	logger.debug('Created Relevant Directories')
	logger.info('Experiment Name: {}'.format(args.run_name))

	vocab_path = os.path.join(args.model_path, 'vocab.p')

	if is_train:
		train_dataloader, val_dataloader = load_data(args, logger)

		logger.debug('Creating Vocab...')

		voc = Syn_Voc()
		voc.create_vocab_dict(args, train_dataloader)

		#Change this as per requirement
		if args.add_val_vocab:
			voc.add_to_vocab_dict(args, val_dataloader)

		logger.info('Vocab Created with number of words : {}'.format(voc.nwords))
		
		with open(vocab_path, 'wb') as f:
			pickle.dump(voc, f, protocol=pickle.HIGHEST_PROTOCOL)

		logger.info('Vocab saved at {}'.format(vocab_path))

	else:
		test_dataloader = load_data(args, logger)
		logger.info('Loading Vocab File...')

		with open(vocab_path, 'rb') as f:
			voc = pickle.load(f)

		logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab_path, voc.nwords))

		# TO DO : Load Existing Checkpoints here
	# checkpoint = get_latest_checkpoint(args.model_path, logger)

	'''Param Specs'''
	layers = args.layers
	heads= args.heads
	d_model = args.d_model
	d_ff = args.d_ff
	max_len = args.max_length
	dropout = args.dropout
	BATCH_SIZE = args.batch_size
	epochs= args.epochs

	if logging_var:
		meta_fname= os.path.join(args.log_path, 'meta.txt')
		loss_fname= os.path.join(args.log_path, 'loss.txt')

		meta_fh = open(meta_fname, 'w')
		loss_fh = open(loss_fname, 'w')

		print('Log Files created at: {}'.format(args.log_path))

		write_meta(args, meta_fh)

	pad_idx = voc.w2id['PAD']

	model = make_model(args, voc.nwords, voc.nwords, N=layers, h=heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
	model.to(device)

	logger.info('Initialized Model')

	criterion = LabelSmoothing(size=voc.nwords, padding_idx=pad_idx, smoothing=0.1)
	criterion.to(device)

	if mode=='train':
		model_opt = NoamOpt(model.src_embed[0].d_model, 1, 3000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
		max_bleu_score = 0.0
		min_error_score = 100.0
		epoch_offset= 0
		logger.info('Starting Training Procedure')
		for epoch in range(epochs):
			print('Training Epoch: ', epoch)
			model.train()
			start_time = time.time()
			run_epoch((rebatch(args, device, voc, voc, pad_idx, b) for b in train_dataloader), 
					  model, 
					  LossCompute(model.generator, criterion, device=device, opt=model_opt))

			time_taken = (time.time() - start_time)/60.0
			logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
			logger.debug('Starting Validation')

			model.eval()

			refs = []
			hyps = []
			error_score = 0

			for i, batch in enumerate(val_dataloader):
				sent1s = sents_to_idx(voc, batch['src'], args.max_length)
				sent2s = sents_to_idx(voc, batch['trg'], args.max_length)
				sent1_var, sent2_var, input_len1, input_len2  = process_batch(sent1s, sent2s, voc, voc, device, voc.id2w[pad_idx])

				sent1s = idx_to_sents(voc, sent1_var, no_eos= True)
				sent2s = idx_to_sents(voc, sent2_var, no_eos= True)

				#for eg in range(sent1_var.size(0)):
				src = sent1_var.transpose(0, 1)
				src_mask = (src != voc.w2id['PAD']).unsqueeze(-2)
				
				# refs.append([' '.join(sent2s[eg])])
				# refs += [[' '.join(sent2s[i])] for i in range(sent2_var.size(1))]
				refs += [[x] for x in batch['trg']]

				out = greedy_decode(model, src, src_mask, 
									max_len=max_len, start_symbol=voc.w2id['<s>'], pad=pad_idx)
				
				words = []

				decoded_words = [[] for i in range(out.size(0))]
				ends = []

				for z in range(1, out.size(1)):
					for b in range(len(decoded_words)):
						sym = voc.id2w[out[b, z].item()]
						if b not in ends:
							if sym == "</s>": 
								ends.append(b)
								continue
							decoded_words[b].append(sym)

				with open(args.outputs_path + '/outputs_epoch_' + str(epoch) + '.txt', 'a') as f_out:
					f_out.write('Batch: ' + str(i) + '\n')
					f_out.write('---------------------------------------\n')
					for z in range(len(decoded_words)):
						try:
							f_out.write('Example: ' + str(z) + '\n')
							f_out.write('Source: ' + batch['src'][z] + '\n')
							f_out.write('Target: ' + batch['trg'][z] + '\n')
							f_out.write('Generated: ' + stack_to_string(decoded_words[z]) + '\n' + '\n')
						except:
							logger.warning('Exception: Failed to generate')
							pdb.set_trace()
							break
					f_out.write('---------------------------------------\n')
					f_out.close()
				
				hyps += [' '.join(decoded_words[z]) for z in range(len(decoded_words))]
				# hyps.append(stack_to_string(words))

				if args.ap:
					error_score += cal_score_AP(decoded_words, batch['trg'])
				else:
					error_score += cal_score(decoded_words, batch['trg'])

				for z in range(1, sent2_var.size(0)):
					sym = voc.id2w[sent2_var[z, 0].item()]
					if sym == "</s>": break

			if (error_score/len(val_dataloader)) < min_error_score:
				min_error_score = error_score/len(val_dataloader)

			val_bleu_epoch = bleu_scorer(refs, hyps)

			if max_bleu_score < val_bleu_epoch[0]:
				max_bleu_score = val_bleu_epoch[0]

			logger.info('Epoch: {}  Val bleu: {}'.format(epoch, val_bleu_epoch[0]))
			logger.info('Maximum Blue: {}'.format(max_bleu_score))
			logger.info('Epoch: {}  Val Error: {}'.format(epoch, error_score/len(val_dataloader)))
			logger.info('Minimum Error: {}'.format(min_error_score))

			if epoch%5 ==0:
				ckpt_path = os.path.join(args.model_path, 'model.pt')
				logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
				torch.save(model.state_dict(), ckpt_path)
				print('Model saved at: {}'.format(ckpt_path))

		store_results(args, max_bleu_score, min_error_score)
		logger.info('Scores saved at {}'.format(args.result_path))
			
	else:
		model.load_state_dict(torch.load(args.model_path))
		model.eval()

if __name__ == '__main__':
	main()