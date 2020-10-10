import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import pdb
from src.components.utils import *


def attention(query, key, value, mask=None, dropout=None):
	"Compute 'Scaled Dot Product Attention'"
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	# pdb.set_trace()
	try:
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)
	except:
		pdb.set_trace()
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)



class BiMultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(BiMultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		# self.linears_bi = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask_fwd=None, mask_bi= None):
		"Implements Figure 2"
		if mask_fwd is not None:
			# Same mask applied to all h heads.
			mask_fwd = mask_fwd.unsqueeze(1)
			mask_bi = mask_bi.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query_fwd, key_fwd, value_fwd = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))]

		# query_bi, key_bi, value_bi = \
		# 	[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
		# 	 for l, x in zip(self.linears_bi, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query_fwd, key_fwd, value_fwd, mask=mask_fwd, 
								 dropout=self.dropout)
		# x2, self.attn_bi = attention(query_bi, key_bi, value_bi, mask=mask_bi, 
		# 						 dropout=self.dropout)

		# x = (x+x2)/2
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)
