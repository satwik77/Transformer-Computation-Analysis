import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
import logging
from torch.autograd import Variable
import pdb
from src.components.utils import *
from src.components.encoder import *
from src.components.decoder import *
from src.components.self_attention import *


class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	# def forward(self, src, tgt, src_mask, tgt_mask):
	# 	"Take in and process masked src and target sequences."
	# 	#pdb.set_trace()
	# 	return self.decode(self.encode(src, src_mask),src_mask,
	# 					   tgt, tgt_mask)

	def forward(self, src, tgt, src_mask_enc, src_mask_dec, tgt_mask, src_mask_bi = None):
		"Directional Take in and process masked src and target sequences."
		#pdb.set_trace()
		return self.decode(self.encode(src, src_mask_enc, src_mask_bi),src_mask_dec,
						   tgt, tgt_mask)

	def encode(self, src, src_mask, src_mask_bi):
		# pdb.set_trace()
		return self.encoder(self.src_embed(src), src_mask, src_mask_bi)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
	"Define standard linear + softmax generation step."

	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
	"Helper: Construct a model from hyperparameters."
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	attn_bi = BiMultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	no_position = NoPositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn_bi), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(no_position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(no_position)),
		Generator(d_model, tgt_vocab))

	# This was important from their code.
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)
	return model


def run_epoch(data_iter, model, loss_compute):
	"Standard Training and Logging Function"
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0
	for i, batch in enumerate(data_iter):
		#pdb.set_trace() # batch.src -> [9,8] batch.trg -> [9,10]
		# out = model.forward(batch.src, batch.trg,
		# 					batch.src_mask, batch.trg_mask)

		out = model.forward(batch.src, batch.trg,
							batch.src_mask_enc, batch.src_mask_dec, batch.trg_mask, batch.src_mask_enc_bi)

		#pdb.set_trace()
		#refs += [[' '.join(batch.trg[i])] for i in range(batch.trg.size(0))]

		#pdb.set_trace()

		ntoks = batch.ntokens.float()
		loss = loss_compute(out, batch.trg_y, ntoks)
		total_loss += loss
		total_tokens += ntoks
		tokens += ntoks
		if i % 200 == 1:
			elapsed = time.time() - start + 1e-8
			print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
				  (i, loss / ntoks, tokens/elapsed))
			start = time.time()
			tokens = 0
	return total_loss / total_tokens
