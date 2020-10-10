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


class Decoder(nn.Module):
	"Generic N layer decoder with masking."

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)


class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		# self.sublayer = clones(SublayerConnection(size, dropout), 3)
		self.sublayer1 = SublayerConnection(size, dropout)
		self.sublayer2 = SublayerConnection(size, dropout)
		# self.sublayer2 = SublayerConnection(size, dropout)
		self.sublayer3 = SublayerConnection(size, dropout)

	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory
		
		x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer2(x, lambda x: self.src_attn(x, m, m, src_mask))


		return self.sublayer3(x, self.feed_forward)



# class DecoderLayer(nn.Module):
#     "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)

#     def forward(self, x, memory, src_mask, tgt_mask):
#         m = memory
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
#         x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
#         return self.sublayer[2](x, self.feed_forward)