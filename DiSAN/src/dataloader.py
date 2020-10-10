import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict

class TextDataset(Dataset):
	'''
		Expecting two files, one for each language

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing
						is_train: 

	'''

	def __init__(self, args, data_path='./data/', dataset='X', datatype='train', max_length=50, is_debug=False, is_train=False):
		if datatype=='train':
			if args.mt:
				src_file_path = os.path.join(data_path, dataset, 'train.' + args.src_lang)
				trg_file_path = os.path.join(data_path, dataset, 'train.' + args.trg_lang)
			else:
				file_path = os.path.join(data_path, dataset, 'train.csv')
		elif datatype=='dev':
			if args.mt:
				src_file_path = os.path.join(data_path, dataset, 'valid.' + args.src_lang)
				trg_file_path = os.path.join(data_path, dataset, 'valid.' + args.trg_lang)
			else:
				file_path = os.path.join(data_path, dataset, 'dev.csv')
		else:
			if args.mt:
				src_file_path = os.path.join(data_path, dataset, 'test.' + args.src_lang)
				trg_file_path = os.path.join(data_path, dataset, 'test.' + args.trg_lang)
			else:
				file_path = os.path.join(data_path, dataset, 'test.csv')

		if args.mt:
			src_file = open(src_file_path, 'r')
			trg_file = open(trg_file_path, 'r')
			self.src = src_file.readlines()
			self.trg = trg_file.readlines()
		else:
			file_df= pd.read_csv(file_path)
			self.src= file_df['Source'].values
			self.trg= file_df['Target'].values

		#pdb.set_trace()

		if is_debug:
			self.src = self.src[:5000:500]
			self.trg = self.trg[:5000:500]

		self.max_length = max_length

		all_sents = zip(self.src, self.trg)

		if is_train:
			all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))

		self.src, self.trg = zip(*all_sents)


	def __len__(self):
		return len(self.src)

	def __getitem__(self, idx):
		#ques = self.process_string(self.unicodeToAscii(str(self.ques[idx])))
		#eqn = self.process_string(self.unicodeToAscii(str(self.eqn[idx])))
		src = self.process_string(str(self.src[idx]))
		trg = self.process_string(str(self.trg[idx]))
		return {'src': src, 'trg': trg}

	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		#string = re.sub(r"\'s", " 's", string)
		#string = re.sub(r"\'ve", " 've", string)
		#string = re.sub(r"n\'t", " n't", string)
		#string = re.sub(r"\'re", " 're", string)
		#string = re.sub(r"\'d", " 'd", string)
		#string = re.sub(r"\'ll", " 'll", string)
		#string = re.sub(r",", " , ", string)
		#string = re.sub(r"!", " ! ", string)
		#string = re.sub(r"\(", " ( ", string)
		#string = re.sub(r"\)", " ) ", string)
		#string = re.sub(r"\?", " ? ", string)
		#string = re.sub(r"\s{2,}", " ", string)
		return string